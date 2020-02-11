//
//  OpenCVWrapper.mm
//  RDT Camera
//
//  Created by developer on 27/01/20.
//  Copyright © 2020 IPRD. All rights reserved.
//
#import "ObjectDetectionUtil.hpp"
#import "OpenCVWrapper.h"
#import <opencv2/imgcodecs/ios.h>

static ObjectDetectionUtil objdetut;

static void UIImageToMat(UIImage *image, cv::Mat &mat) {
    assert(image.size.width > 0 && image.size.height > 0);
    assert(image.CGImage != nil || image.CIImage != nil);

    // Create a pixel buffer.
    NSInteger width = image.size.width;
    NSInteger height = image.size.height;
    cv::Mat mat8uc4 = cv::Mat((int)height, (int)width, CV_8UC4);

    // Draw all pixels to the buffer.
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    if (image.CGImage) {
        // Render with using Core Graphics.
        CGContextRef contextRef = CGBitmapContextCreate(mat8uc4.data, mat8uc4.cols, mat8uc4.rows, 8, mat8uc4.step, colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault);
        CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), image.CGImage);
        CGContextRelease(contextRef);
    } else {
        // Render with using Core Image.
        static CIContext* context = nil; // I do not like this declaration contains 'static'. But it is for performance.
        if (!context) {
            context = [CIContext contextWithOptions:@{ kCIContextUseSoftwareRenderer: @NO }];
        }
        CGRect bounds = CGRectMake(0, 0, width, height);
        [context render:image.CIImage toBitmap:mat8uc4.data rowBytes:mat8uc4.step bounds:bounds format:kCIFormatRGBA8 colorSpace:colorSpace];
    }
    CGColorSpaceRelease(colorSpace);

    // Adjust byte order of pixel.
    cv::Mat mat8uc3 = cv::Mat((int)width, (int)height, CV_8UC3);
    cv::cvtColor(mat8uc4, mat8uc3, cv::COLOR_RGBA2BGR);

    mat = mat8uc3;
}

/// Converts a Mat to UIImage.
static UIImage *MatToUIImage(cv::Mat &mat) {

    // Create a pixel buffer.
    assert(mat.elemSize() == 1 || mat.elemSize() == 3);
    cv::Mat matrgb;
    if (mat.elemSize() == 1) {
        cv::cvtColor(mat, matrgb, cv::COLOR_GRAY2RGB);
    } else if (mat.elemSize() == 3) {
        cv::cvtColor(mat, matrgb, cv::COLOR_BGR2RGB);
    }

    // Change a image format.
    NSData *data = [NSData dataWithBytes:matrgb.data length:(matrgb.elemSize() * matrgb.total())];
    CGColorSpaceRef colorSpace;
    if (matrgb.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef imageRef = CGImageCreate(matrgb.cols, matrgb.rows, 8, 8 * matrgb.elemSize(), matrgb.step.p[0], colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault, provider, NULL, false, kCGRenderingIntentDefault);
    UIImage *image = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);

    return image;
}

/// Restore the orientation to image.
static UIImage *RestoreUIImageOrientation(UIImage *processed, UIImage *original) {
    if (processed.imageOrientation == original.imageOrientation) {
        return processed;
    }
    return [UIImage imageWithCGImage:processed.CGImage scale:1.0 orientation:original.imageOrientation];
}

@implementation OpenCVWrapper

+ (NSString *)openCVVersionString {
return [NSString stringWithFormat:@"OpenCV Version %s",  CV_VERSION];
}

+ (CGRect)update:(MLMultiArray *)rdtOutput :(bool[])RDT{
    CGRect roi = CGRectMake(-1.0, -1.0, -1.0,-1.0);
	
    return roi;
    
}
+ (NSArray*)getRGBAsFromImage:(UIImage*)image atX:(int)x andY:(int)y count:(int)count
{
    NSMutableArray *result = [NSMutableArray arrayWithCapacity:count];

    // First get the image into your data buffer
    CGImageRef imageRef = [image CGImage];
    NSUInteger width = CGImageGetWidth(imageRef);
    NSUInteger height = CGImageGetHeight(imageRef);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    unsigned char *rawData = (unsigned char*) calloc(height * width * 4, sizeof(unsigned char));
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    CGContextRef context = CGBitmapContextCreate(rawData, width, height,
                    bitsPerComponent, bytesPerRow, colorSpace,
                    kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);

    CGContextDrawImage(context, CGRectMake(0, 0, width, height), imageRef);
    CGContextRelease(context);

    // Now your rawData contains the image data in the RGBA8888 pixel format.
    NSUInteger byteIndex = (bytesPerRow * y) + x * bytesPerPixel;
    for (int i = 0 ; i < count ; ++i)
    {
        CGFloat alpha = ((CGFloat) rawData[byteIndex + 3] ) / 255.0f;
        CGFloat red   = ((CGFloat) rawData[byteIndex]     ) / alpha;
        CGFloat green = ((CGFloat) rawData[byteIndex + 1] ) / alpha;
        CGFloat blue  = ((CGFloat) rawData[byteIndex + 2] ) / alpha;
        byteIndex += bytesPerPixel;

        UIColor *acolor = [UIColor colorWithRed:red green:green blue:blue alpha:alpha];
        [result addObject:acolor];
    }

  free(rawData);

  return result;
}

+ (double)detect2wrapper:(CGPoint)arrowPreds :(CGPoint)Cpreds :(CGPoint)InfPreds :(const double[])orientations :(CGPoint *)outscalerot{
    
    CvPoint2D32f a(arrowPreds.x,arrowPreds.y);
    CvPoint2D32f c(Cpreds.x,Cpreds.y);
    CvPoint2D32f i(InfPreds.x,InfPreds.y);
    CvPoint2D32f out_scale_rot(outscalerot->x,outscalerot->y);

    CvPoint3D32f orients;
    orients.x = orientations[0];    orients.y = orientations[1];    orients.z = orientations[2];
    
//    a.x = float(arrowPreds.x);

    
    double err = objdetut.detect2(a, c, i, orients, &out_scale_rot);
    outscalerot->x=out_scale_rot.x;outscalerot->y=out_scale_rot.y;

    return err;
}

+ (nonnull UIImage *)preprocessImage:(nonnull UIImage *)image {
    cv::Mat bgrMat;
    UIImageToMat(image, bgrMat);
    cv::Mat grayMat;
    cv::Mat pyrDown1;
    cv::Mat pyrDown2;
    cv::cvtColor(bgrMat, grayMat, cv::COLOR_BGR2GRAY);
    cv::pyrDown(grayMat, pyrDown1);
    cv::pyrDown(pyrDown1, pyrDown2);
    if(pyrlevelcnt==1){
        UIImage *grayImage = MatToUIImage(pyrDown1);
        return RestoreUIImageOrientation(grayImage, image);

    }
    else if (pyrlevelcnt==2){
        UIImage *grayImage = MatToUIImage(pyrDown2);
        return RestoreUIImageOrientation(grayImage, image);

    }
    else{
        UIImage *grayImage = MatToUIImage(grayMat);
        return RestoreUIImageOrientation(grayImage, image);

    }
}

+ (CGRect)returnBoundingRect:(CGPoint)rdt :(double)s_w :(double)s_h :(double)calculatedAngle {
    
    CGRect tmp = CGRectMake(-1.0, -1.0, -1.0, -1.0);
    cv::Size sz(s_w,s_h);
    cv::Point cent(rdt.x,rdt.y);
    cv::RotatedRect rotrec(cent,sz,calculatedAngle);
    cv::Rect tmp1 =rotrec.boundingRect();
    tmp.origin.x = tmp1.x;
    tmp.origin.y = tmp1.y;
    tmp.size.width = tmp1.width;
    tmp.size.height = tmp1.height;
    return tmp;
}

+(int)checkSteadyStatus:(UIImage*)inp{
    cv::Mat colMat;
    cv::Mat greyMat;
    
    UIImageToMat(inp, colMat);
    cv::cvtColor(colMat, greyMat, cv::COLOR_BGR2GRAY);
    int res = objdetut.checkSteady(greyMat);
    return res;
}

@end
