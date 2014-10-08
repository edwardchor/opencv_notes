#include <iostream>
#include <opencv.hpp>
using namespace cv;
using namespace std;
class ColorHistogram{
    private:
    int histSize[3];
    float hranges[2];
    const float*ranges[3];
    int channels[3];
    public:
    ColorHistogram(){
        histSize[0]=histSize[1]=histSize[2]=256;
        hranges[0]=0.0;
        hranges[1]=255.0;
        ranges[0]=hranges;
        ranges[1]=hranges;
        ranges[2]=hranges;
        channels[0]=0;
        channels[1]=1;
        channels[2]=2;
    }
    SparseMat getSparseHistogram(const Mat &image){
        SparseMat hist(3,histSize,CV_32F);
        calcHist(&image,1,channels,Mat(),hist,3,histSize,ranges);
        return hist;
    }
    void colorReduceAt(Mat &image,int div=64){
        int nl=image.rows;
        int nc=image.cols;
        for(int i=0;i<nl;i++){
            for(int j=0;j<nc;j++){
                image.at<Vec3b>(i,j)[0]=image.at<Vec3b>(i,j)[0]/div*div+div/2;
                image.at<Vec3b>(i,j)[1]=image.at<Vec3b>(i,j)[1]/div*div+div/2;
                image.at<Vec3b>(i,j)[2]=image.at<Vec3b>(i,j)[2]/div*div+div/2;
            }

        }
    }
    MatND getHistogram(const Mat& image){
        MatND hist;
        calcHist(&image,1,channels,Mat(),hist,3,histSize,ranges);
        return hist;
    }
};
class Histogram1D{
    private:
    int histSize[1];
    float hranges[2];
    const float*ranges[1];
    int channels[1];
    public:
    Histogram1D(){
        histSize[0]=256;
        hranges[0]=0.0;
        hranges[1]=255.0;
        ranges[0]=hranges;
        channels[0]=0;
    }
    MatND getHistogram(const cv::Mat &image){
        MatND hist;
        calcHist(&image,1,channels,Mat(),hist,1,histSize,ranges);
        return hist;
    }
    Mat getHistogramImage(const cv::Mat &image){
        Mat hist=getHistogram(image);
        double maxVal=0;
        double minVal=0;
        minMaxLoc(hist,&minVal,&maxVal,0,0);
        Mat histImg(histSize[0],histSize[0],CV_8U,cv::Scalar(255));
        int hpt=static_cast<int>(0.9*histSize[0]);
        for(int h=0;h<histSize[0];h++){
            float binVal=hist.at<float>(h);
            int intensity=static_cast<int>(binVal*hpt/maxVal);
            line(histImg,Point(h,histSize[0]),Point(h,histSize[0]-intensity),Scalar::all(0));
        }
        return histImg;
    }
    Mat applyLookUp(const Mat &image,const Mat& lookup){
        Mat result;
        LUT(image,lookup,result);
        return result;
    }
    Mat stretch(const Mat &image,int minValue=0){
        MatND hist=getHistogram(image);
        int imin=0;
        for(;imin<histSize[0];imin++){
            cout<<hist.at<float>(imin)<<endl;
            if(hist.at<float>(imin)>minValue)
            break;
        }
     int imax=histSize[0]-1;
     for(;imax;imax--){
         if(hist.at<float>(imax)>minValue)
             break;
         }
        int dim(256);
        Mat lookup(1,&dim,CV_8U);
        for(int i=0;i<256;i++){
            if(i<imin)lookup.at<uchar>(i)=0;
            else if(i>imax)lookup.at<uchar>(i)=255;
            else lookup.at<uchar>(i)=static_cast<uchar>(255.0*(i-imin)/(imax-imin)+0.5);
        }
        Mat result;
        result=applyLookUp(image,lookup);
        return result;
    }
    void sharpen(const Mat &image,Mat &result){
        int c=image.channels();
        int nc=image.cols*image.channels();
        int nl=image.rows;
        result.create(image.size(),image.type());
        for(int j=1;j<nl-1;j++){
            const uchar*previous=image.ptr<const uchar>(j-1);
            const uchar* current=image.ptr<const uchar>(j);
            const uchar* next=image.ptr<const uchar>(j+1);
            uchar* output=result.ptr<uchar>(j);
            for(int i=c;i<nc-c;i++){
                *output++=cv::saturate_cast<uchar>(
                            5*current[i]-current[i-c]-current[i+c]-previous[i]-next[i]);
            }
        }
        result.row(0).setTo(cv::Scalar(0));
        result.row(result.rows-1).setTo(cv::Scalar(0));
        result.col(0).setTo(cv::Scalar(0));
        result.col(result.cols-1).setTo(cv::Scalar(0));
    }
};
class ContentFinder{
    private:
    float hranges[2];
    const float*ranges[3];
    int channels[3];
    float threshold;
    MatND histogram;
    public:
    ContentFinder():threshold(-1.0f){
        ranges[0]=hranges;
        ranges[1]=hranges;
        ranges[2]=hranges;
    }
    void setThreshold(float t){
        threshold=t;
    }
    float getThreshold(){
        return threshold;
    }
    void setHistogram(const MatND &h){
        histogram=h;
        normalize(histogram,histogram,1.0);
    }
    Mat find(const Mat& image,float minValue,float maxValue,int* channels,int dim){
        Mat result;
        hranges[0]=minValue;
        hranges[1]=maxValue;
        for(int i=0;i<dim;i++){
            this->channels[i]=channels[i];
            calcBackProject(&image,1,channels,histogram,result,ranges,255.0);
        }
        if(threshold>0.0)
           cv:: threshold(result,result,255*threshold,255,THRESH_BINARY);
        return result;
    }

};
int main()
{
  Mat cloud;
  cloud=imread("/home/edwardchor/pictures/cloud.jpg",0);
  Histogram1D h;
  Mat cloudHistImage= h.getHistogramImage(cloud);
        imshow("Cloud Hist",cloudHistImage);
  Mat cloudROI;
        cloudROI=cloud(Rect(560,125,40,50));
  MatND hist=h.getHistogram(cloudROI);
        normalize(hist,hist,1.0);
  Mat result;
        h.calcBP(cloud,result,hist);
        imshow("Origin Cloud",cloud);
  ColorHistogram hc;
  Mat color=imread("/home/edwardchor/pictures/cloud.jpg");
        hc.colorReduceAt(color,32);
  Mat imageROI=color(Rect(0,0,165,75));
  MatND colorhist=hc.getHistogram(imageROI);
  ContentFinder finder;
        finder.setHistogram(colorhist);
        finder.setThreshold(0.05f);
  Mat colorresult=finder.find(color);
        waitKey(0);
    return 0;
}

