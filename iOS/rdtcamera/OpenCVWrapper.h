//
//  OpenCVWrapper.h
//  RDT Camera
//
//  Created by developer on 27/01/20.
//  Copyright © 2020 IPRD. All rights reserved.
//

#import <CoreGraphics/CoreGraphics.h>
#import <CoreML/CoreML.h>
#import <UIKit/UIKit.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

static float deviationThresh=0.01f;
//if(mModelFileName.contains("360x640"))
//
//int inputSize[]={360,640};
//int aspectAnchors[]={30, 70, 68, 68, 44, 74, 28, 52};
//int numberBlocks[]={10,19};
//int pyrlevelcnt =1;

//if(mModelFileName.contains("180x320"))
//
//int inputSize[]= {180,320};
//int aspectAnchors[]={15, 35, 34,34, 22, 37, 14, 26};
//int numberBlocks[]= {5, 9};
//int pyrlevelcnt =2;

//if(mModelFileName.contains("180x320") and "new_arch")

static int inputSize[]= {180,320};
static int numberBlocks[]= {7, 16};
static int pyrlevelcnt =2;
static int numberClasses = 31;
static double mThreshold = 0.9;

@interface OpenCVWrapper : NSObject


+ (NSString *)openCVVersionString;
+ (CGRect)update:(MLMultiArray *)rdtOutput :(bool[])RDT;
+ (UIImage *)preprocessImage:(UIImage *)image;
+ (NSArray*)getRGBAsFromImage:(UIImage*)image atX:(int)x andY:(int)y count:(int)count;
+ (double)detect2wrapper:(CGPoint)arrowPreds :(CGPoint)Cpreds :(CGPoint)InfPreds :(const double[])orientations :(CGPoint *)outscalerot;
+ (CGRect)returnBoundingRect:(CGPoint)rdt :(double)s_w :(double)s_h :(double)calculatedAngle;
+ (int)checkSteadyStatus:(UIImage*)inp;
@end

NS_ASSUME_NONNULL_END

