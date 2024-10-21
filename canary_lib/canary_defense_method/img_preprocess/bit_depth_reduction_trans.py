class BitDepthReduction(object):
    def __init__(self, device, compressed_bit=4):
        self.compressed_bit = compressed_bit
        self.device = device
    def bit_depth_reduction(self, xs):
        bits = 2 ** self.compressed_bit #2**i
        xs_compress = (xs.detach() * bits).int()
        xs_255 = (xs_compress * (255 / bits))
        xs_compress = (xs_255 / 255).to(self.device)

        return xs_compress

    def forward(self, x):
        return self.bit_depth_reduction(x)
