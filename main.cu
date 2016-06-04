#include <iostream>
#include "test/test.hpp"

int main(void) {
	determinant_tests();
	point_set_tests();
	std::cout << "Tests completed successfully!" << std::endl;
	return 0;
}
