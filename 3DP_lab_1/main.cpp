#include "sgm.h"

int main(int argc, char** argv) {

    if (argc != 7) {
        cerr << "Usage: " << argv[0] << " <right image> <left image> <monocular_right>  <gt disparity map> <output image file> <disparity range> " << endl;
        return -1;
    }

    char *firstFileName = argv[1];
    char *secondFileName = argv[2];
    char *monoRightFileName = argv[3];
    char *gtFileName = argv[4];
    char *outputFileName = argv[5];
    unsigned int disparityRange = atoi(argv[6]);

    cv::Mat firstImage;
    cv::Mat secondImage;
    cv::Mat monoRight;
    cv::Mat gt;

    firstImage = cv::imread(firstFileName, IMREAD_GRAYSCALE);
    secondImage = cv::imread(secondFileName, IMREAD_GRAYSCALE);
    monoRight = cv::imread(monoRightFileName, IMREAD_GRAYSCALE);
    gt = cv::imread(gtFileName, IMREAD_GRAYSCALE);

    if(!firstImage.data || !secondImage.data) {
        cerr <<  "Could not open or find one of the images!" << endl;
        return -1;
    }



    sgm::SGM sgm(disparityRange);
    sgm.set(firstImage, secondImage, monoRight);
    sgm.compute_disparity();
    sgm.save_disparity(outputFileName);
    //sgm.save_confidence(outputFileName);
    std::cerr<<"Right Image MSE error: "<<sgm.compute_mse(gt)<<std::endl;

    return 0;
}
