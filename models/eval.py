#to extract features from inception, we need to rebuild the model.
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import inception_v3
import numpy as np
import scipy

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0.0, 1.0)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
    

#define FID
#A and B are two 2048 size vectors
def calculate_FID_distance(A, B):
    mu_A = np.mean(A)
    mu_B = np.mean(B)
    cov_A = np.cov(A)
    cov_B = np.cov(B)
    covmean = scipy.linalg.sqrtm(cov_A.dot(cov_B))
    ssq = np.sum(np.square(mu_A-mu_B))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    return ssq + np.trace(cov_A+cov_B-2*covmean)

def calculate_dij(real_vec, gen_vec):
    return 1 - np.dot(real_vec, gen_vec)/(np.linalg.norm(real_vec)*np.linalg.norm(gen_vec))

def calculate_memorization(gen_matrix, real_matrix, eps):
    #dimension of real: M*2048, gen: N*2048
    #final score matrix: N*M (gen by real)
    print(gen_matrix)
    score = np.zeros((gen_matrix.shape[0], real_matrix.shape[0]))
    #e.g 3 by 2048
    for i in range(gen_matrix.shape[0]):
        for j in range(real_matrix.shape[0]):
            score[i,j] = calculate_dij(real_matrix[j,:], gen_matrix[i,:])
    
    #average all the generated images
    d = np.mean(np.min(score, axis=0))
    if d < eps:
        dthr = d
    else:
        dthr = 1
    
    return dthr
    
def get_mifid(gen_matrix, real_matrix, eps = 1e-15):
    d = calculate_FID_distance(gen_matrix, real_matrix)
    mem = calculate_memorization(gen_matrix, real_matrix, eps)
    return d/mem


#run this to get mifid score
def get_mifid_from_images(gen_img, real_img, eps = 1e-15):
    v3 = torch.load('inceptionv3.pkl')
    gen_matrix = v3(gen_img).squeeze().numpy()
    real_img = v3(real_img).squeeze().numpy()
    return get_mifid(gen_matrix, real_matrix)
    