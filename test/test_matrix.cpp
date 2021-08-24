#include <Matrix/Matrix.hpp>
#include <functions/common.hpp>

using namespace snn;

int main()
{
    float a1[9] = {0, 2, 3, 4, 5, 6, 7, 8, 6};
    float a2[9] = {2, 4, 6, 8, 1, 3, 5, 7, 9};
    
    Matrix_f m1(a1, 3, 3);
    Matrix_f m2(a2, 3, 3);

    

    std::cout << "m1 = " << std::endl;
    std::cout << m1 << std::endl;
    std::cout << "rank(m1) = " << m1.getRank() << std::endl;
    std::cout << "det(m1) = " << m1.det() << std::endl;
    std::cout << "inv(m1) = " << m1.inv() << std::endl;
    std::cout << "det(inv(m1)) = " << m1.inv().det() << std::endl;
    /*
    std::cout << "m2 = " << std::endl;
    std::cout << m2 << std::endl;
    std::cout << "m1 + m2 = " << std::endl;
    std::cout << m1 + m2 << std::endl;
    std::cout << "m1 - m2 = " << std::endl;
    std::cout << m1 - m2 << std::endl;
    std::cout << "m1 * m2 = " << std::endl;
    std::cout << m1 * m2 << std::endl;
    std::cout << "m2 * m1 = " << std::endl;
    std::cout << m2 * m1 << std::endl;

    Matrix<float> m3 = m1.sub_matrix(0, 1, 0, 2);
    std::cout << "m3 = " << std::endl;
    std::cout << m3 << std::endl;
    std::vector<int> rset {0, 2};
    std::vector<int> cset {2, 1};
    m3 = m1.sub_matrix(rset, cset);
    std::cout << "m3 = " << std::endl;
    std::cout << m3 << std::endl;
    m3 = m1.transpose();
    std::cout << "m3 = " << std::endl;
    std::cout << m3 << std::endl;

    Matrix<int> m4 = m3.cast<int>();
    std::cout << "m4 = " << std::endl;
    std::cout << m4 << std::endl;
    Matrix<char> m5 = m4.cast<char>();
    std::cout << "m5 = " << std::endl;
    std::cout << m5 << std::endl;
    Matrix<float> m6 = eye<float>(5);
    std::cout << "m6 = " << std::endl;
    std::cout << m6 << std::endl;*/

    float a7[16] = {0, 2, 3, 4, 5, 6, 7, 8, 6, 2, 5, 10, 2, 4, 6, 6};
    Matrix_f m7(a7, 4, 4);

    std::cout << "m7 = " << std::endl;
    std::cout << m7 << std::endl;
    std::cout << "rank(m7) = " << m7.getRank() << std::endl;
    std::cout << "det(m7) = " << m7.det() << std::endl;
    std::cout << "inv(m7) = " << m7.inv() << std::endl;
    std::cout << "det(inv(m7)) = " << m7.inv().det() << std::endl;
    std::cout << "m7 * inv(m7) = " << m7 * (m7.inv()) << std::endl;
    std::cout << "m7 * 0.6 = " << m7 * 0.6 << std::endl;
    std::cout << "m7 + 2 = " << m7 + 2 << std::endl;
    m7 -= m7 * 0.6;
    std::cout << "m7 -= (m7 * 0.6) = " << m7 << std::endl;
    std::cout << "m7 ^ 2 = " << m7.element_pow(2.0) << std::endl; 
    
    return 0;
}
