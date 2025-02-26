#include <iostream>
#include "../../libtorch/include/torch/csrc/api/include/torch/torch.h"

namespace midigptTorch {
	void testMmmTorch() {
		std::cout << "midigptTorch test function called" << std::endl;
		torch::Tensor tensor = torch::rand({ 2, 3 });
		std::cout << tensor << std::endl;
	}
}