//
//  ObjectDetection.swift
//  RDT Camera
//
//  Created by developer on 27/01/20.
//  Copyright © 2020 IPRD. All rights reserved.
//

import Foundation

import UIKit
import CoreML

@available(iOS 13.0, *)

struct acceptanceStatus {
   var mScale: Float
   var mBrightness: Int
    var RDT_found: Bool

   init(mScale: Float, mBrightness: Int, RDT_found: Bool) {
      self.mScale = mScale
      self.mBrightness = mBrightness
      self.RDT_found = RDT_found
   }
}
func cropImage(image: UIImage, rect: CGRect) -> UIImage {
    let cgImage = image.cgImage! // better to write "guard" in realm app
    let croppedCGImage = cgImage.cropping(to: rect)
    return UIImage(cgImage: croppedCGImage!)
}
func deg2rad(_ number: Double) -> Double {
    return number * .pi / 180
}

func rad2deg(_ number: Double) -> Double {
    return number * 180 / .pi
}

struct resultObj {
    var conf:Double
    var cx:Double
    var cy:Double
    var w:Double
    var h:Double
    var predictedOrientation:Double
}
extension UIColor {
    var components: (red: CGFloat, green: CGFloat, blue: CGFloat, alpha: CGFloat)? {
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        return getRed(&r, green: &g, blue: &b, alpha: &a) ? (r,g,b,a) : nil
    }
}



class ObjectDetection {
     var found = false;
    var widthFactor:Double// = 1.0/Double(inputSize.1)*896.0;
    var heightFactor:Double// = 1.0/Double(inputSize.0)*414.0;
    let rdtModel = rdt();
    var currentImageFrame: UIImage;
    var output: MLMultiArray ;
    var roi: CGRect ;
    var resizeFactor: [Double] ;
    var aspectAnchors: [Double];
    var orientationAngles: [Double];
    var numberAnchors:Int;
    var A_C_to_L = 1.624579124579125;
    var L_to_W = 0.0601036269430052;
    var ref_hyp = 35.0;
    var minError = 100.0;
    var ac_can = 594.0;
    init(W: Double, H: Double) {
        self.widthFactor =  1.0/Double(inputSize.1)*H; // Image view is rotated
        self.heightFactor = 1.0/Double(inputSize.0)*W;
         currentImageFrame = UIImage.init() ;
         output = MLMultiArray.init() ;
         roi = CGRect.init(x: -1.0, y: -1.0, width: -1.0, height: -1.0);
         resizeFactor = [Double(inputSize.0)/Double(numberBlocks.0),Double(inputSize.1)/Double(numberBlocks.1)];
         aspectAnchors=[15.0, 35.0, 34.0,34.0, 22.0, 37.0, 14.0, 26.0];
         orientationAngles=[0,22.5,45,135,157.5,180,202.5,225,315,337.5];
         numberAnchors=aspectAnchors.count/2;
        
    }
    
    
    

    
    
    
    let imageMetadataCustom = [
        "UUID": UUID().uuidString,
        "Quality_parameters": [
            "brightness":"10"
        ],
        "RDT_Type":"Flu_Audere",
        "Include_Proof": "True",
    ] as [String : Any];


    public func  update(imageFrame: UIImage, RDT:inout acceptanceStatus) -> CGRect{
        //currentImageFrame = UIImage.init(cgImage: UIImage() as! CGImage);
        currentImageFrame = OpenCVWrapper.preprocessImage(imageFrame);
//        UIImageWriteToSavedPhotosAlbum(currentImageFrame,nil,nil,nil);
        minError = 100.0;
        output = processROIDetection(imageFrame: currentImageFrame);
        let curr_time = NSDate().timeIntervalSince1970;

        var vectorTableArrow = [resultObj]() //alternatively (does the same): var array = Array<Country>()
        var vectorTableCpattern = [resultObj]() //alternatively (does the same): var array = Array<Country>()
        var vectorTableInfluenza = [resultObj]() //alternatively (does the same): var array = Array<Country>()


        for row in 0...numberBlocks.0-1 {
            for col in 0...numberBlocks.1-1 {
                for j in 0...numberAnchors-1{
                    let computedIndex = row * numberBlocks.1 + col;
                    let targetClass = Argmax(arrayObj: output,row: NSNumber(value: computedIndex),col: NSNumber(value: j),start: 0,end: 31); // Arrays.copyOfRange(output[0][computedIndex][j],0,31));
                    var index: [NSNumber] = [0, NSNumber(value: computedIndex), NSNumber(value: j),NSNumber(integerLiteral: targetClass)];

                    var tmp=0;
                    let confidence=Double(truncating: output[index]);
                    if (confidence>mThreshold) {
                        let offsetStartIndex = numberClasses;
                        index = [0,NSNumber(value: computedIndex),NSNumber(value: j),NSNumber(value: offsetStartIndex)];
                        var cx = (Double(col) + 0.5) *
                                resizeFactor[1] +
                            Double(truncating: output[index]) *
                                Double(inputSize.1);
                        
                        cx = cx*widthFactor;

                        index = [0,NSNumber(value: computedIndex),NSNumber(value: j),NSNumber(value: offsetStartIndex+1)];

                        var cy = (Double(row) + 0.5) *
                            resizeFactor[0] +
                            Double(truncating: output[index]) *
                            Double(inputSize.0);
                        
                        cy = cy*heightFactor;
                        
   
                        
                        index = [0,NSNumber(value: computedIndex),NSNumber(value: j),NSNumber(value: offsetStartIndex+2)];

                        tmp=Int(j*2+1);
                        
                        let w = (aspectAnchors[tmp] * exp(Double(truncating: output[index]))) * widthFactor;

                        tmp=Int(j*2);
                        
                        index = [0,NSNumber(value: computedIndex),NSNumber(value: j),NSNumber(value: offsetStartIndex+3)];

                        let h = (aspectAnchors[tmp] * exp(Double(truncating: output[index]))) * heightFactor;
                        let typeOfFeat=targetClass/10;
                        let predictedOrientation =  orientationAngles[targetClass % 10];
                        let tmp = resultObj(conf: confidence, cx: cx, cy: cy, w: w, h: h, predictedOrientation: predictedOrientation)
                        if (typeOfFeat==2){
                            vectorTableArrow.append(tmp);
                        }
                        else if (typeOfFeat==1){
                            vectorTableCpattern.append(tmp);
                        }
                        else if (typeOfFeat==0){
                            vectorTableInfluenza.append(tmp);
                        }
                    }
                    
                }
            }
        }
//        print(vectorTableArrow.count,vectorTableCpattern.count,vectorTableInfluenza.count);
        if (vectorTableArrow.count > 0) {
            vectorTableArrow.sorted(by: { $0.conf > $1.conf })

        }
        if (vectorTableCpattern.count > 0) {
            vectorTableCpattern.sorted(by: { $0.conf > $1.conf })

        }
        if (vectorTableInfluenza.count > 0) {
            vectorTableInfluenza.sorted(by: { $0.conf > $1.conf })

        }
        if (vectorTableArrow.count > 0 && vectorTableCpattern.count > 0 && vectorTableInfluenza.count > 0) {
            roi = locateRdt(vecArr: vectorTableArrow, vecC: vectorTableCpattern,vecInf: vectorTableInfluenza,rdtRes:&RDT);
            RDT.RDT_found = found;
            let cropepdImg = cropImage(image: imageFrame,rect: roi);
            computeBrightness(inp: cropepdImg, ret: &RDT);

        }
        else{
            RDT.RDT_found = false;

        }
        
//        print("Time taken for post processing",NSDate().timeIntervalSince1970-curr_time)

        return roi;
    }
    
    
    private func locateRdt(vecArr:Array<resultObj>,vecC:Array<resultObj>,vecInf:Array<resultObj>,rdtRes:inout acceptanceStatus)->CGRect{
        
        var roiIn = CGRect(x: -1, y: -1, width: -1, height: -1);
        found = false;
        var C_arrow_best = CGPoint(x: 0.0, y: 0.0);
        var C_Cpattern_best=CGPoint(x: 0.0, y: 0.0);
        var C_infl_best=CGPoint(x: 0.0, y: 0.0);
        var best_scale_rot=CGPoint(x: 0.0, y: 0.0);
        var scale_rot=CGPoint(x:0.0,y:0.0);
        for resArr in vecArr{
            for resC in vecC{
                for resI in vecInf{
                    
                    let arrowP = CGPoint(x:resArr.cx,y:resArr.cy);
                    let CpattP = CGPoint(x:resC.cx,y:resC.cy);
                    let InfP = CGPoint(x:resI.cx,y:resI.cy);
                
                    var orients = [resArr.predictedOrientation,resC.predictedOrientation,resI.predictedOrientation];
                    let tmperror = OpenCVWrapper.detect2wrapper(arrowP, CpattP, InfP, orients, &scale_rot);
                    //detect2(C_arrow, C_Cpattern, C_Inlf,scale_rot);

                    if (tmperror<minError) {
                            minError=tmperror;
                            found = true;
                            C_arrow_best = arrowP;
                            C_Cpattern_best = CpattP;
                            C_infl_best = InfP;
                            best_scale_rot=scale_rot;
                    }
                }
                
            }
            
            
        }
        rdtRes.mScale=Float(best_scale_rot.x);
        var angleRads=Double(best_scale_rot.y);
        if(angleRads>Double.pi){
            angleRads=angleRads-Double.pi*2;
        }
          
    
        let calculatedAngleRotation=rad2deg(angleRads);
        _ = calculatedAngleRotation;
        let tmpcx = C_arrow_best.x+(C_Cpattern_best.x-C_arrow_best.x)/2;
        let tmpcy = C_arrow_best.y+(C_Cpattern_best.y-C_arrow_best.y)/2;
        var rdt_c = CGPoint(x: tmpcx, y: tmpcy)
        _ = rdt_c;
        
        let tmpW = ac_can * A_C_to_L * Double(best_scale_rot.x);
        let tmpH = tmpW * L_to_W;

        if(true) {
          rdt_c.x = rdt_c.x + CGFloat(ref_hyp * cos(angleRads));
          rdt_c.y = rdt_c.y - CGFloat(ref_hyp * sin(angleRads));
        }
        
        
        
//        RotatedRect rotatedRect = new RotatedRect(rdt_c, sz, calculatedAngleRotation);
        roiIn = OpenCVWrapper.returnBoundingRect(rdt_c,tmpW,tmpH,calculatedAngleRotation);
        
        //if(tmp_for_draw != null) putText(tmp_for_draw, "MinError = " + String.valueOf(minError), new Point(0, tmp_for_draw.cols()>>1), FONT_HERSHEY_SIMPLEX, 1.5,new Scalar(255,0,0,0),2);
        //Log.d("ROI:", "X : " + roi.x + "Y : " + roi.y + "W : " + roi.width + "H : " + roi.height);
        return roiIn;
    }
    
    private func Argmax(arrayObj:MLMultiArray,row:NSNumber,col:NSNumber,start:Int,end:Int)->Int{
        var argmax=0;
        var maxval = 0.0;
        for j in start...end {

            let index: [NSNumber] = [0, row, col,NSNumber(integerLiteral: j)];

            let value = Double(truncating: arrayObj[index])

            if (value>maxval){
                maxval=value;
                argmax=j;
            }
        }
        
        
        return argmax
    }
    private func processROIDetection(imageFrame: UIImage) -> MLMultiArray{
        
        let imageFrameCG = imageFrame.cgImage;
        let w = imageFrameCG?.width;
        let h = imageFrameCG?.height;
        let c = w!*h!;
        var curr_time = NSDate().timeIntervalSince1970
        let pixVal = OpenCVWrapper.getRGBAsFrom(imageFrame, atX: 0, andY: 0, count: Int32(c));
//        print("Time taken to extract RGB values",NSDate().timeIntervalSince1970-curr_time)
        guard let mlArray = try? MLMultiArray(shape:[1,180,320,1],
                                                   dataType:MLMultiArrayDataType.float32) else {
                                                    fatalError("MLMultiArray conversion error for image")
        }
        var normalizedPixel=0.0;
        curr_time = NSDate().timeIntervalSince1970
        if let pixelVals = pixVal as NSArray as? [UIColor] {
            for row in 0...h!-1{
                for col in 0...w!-1{
                    let pix_ind=row*w! + col;
                    let index:[NSNumber]=[0,NSNumber(value:179-row),NSNumber(value:col),0];
                    if let myColorComponents = pixelVals[pix_ind].components {
                        normalizedPixel=Double(myColorComponents.green)/255.0;
                        mlArray[index]=NSNumber(value: normalizedPixel);
                        }
                    mlArray[index]=NSNumber(value: normalizedPixel);
                }
            }
  
        }
//        print("Time taken to fill ML array",NSDate().timeIntervalSince1970-curr_time)
        curr_time = NSDate().timeIntervalSince1970


        guard let predResult = try? rdtModel.prediction(input_1: mlArray) else {
            fatalError("rdt model error")
        }
//        print("Time taken to predict",NSDate().timeIntervalSince1970-curr_time)

        return predResult.Identity;
    }
    
    private func computeBrightness(inp: UIImage,ret:inout acceptanceStatus)->Bool {
        //        if(false) {
//           //Log.d("Brightness","mBrightness "+brightness);
//           if (brightness > mConfig.mMaxBrightness) {
//               ret.mBrightness = TOO_HIGH;
//               return false;
//           } else if (brightness < mConfig.mMinBrightness) {
//               ret.mBrightness = TOO_LOW;
//               return false;
//           }
//        }
        ret.mBrightness =  Int(OpenCVWrapper.checkBrightness(inp));
        return true;
    }
    
    public func evaluateRDTRestApi(callback: @escaping (String) -> ()){
        // call this on button click from UI
        let paramName = Utils.dict2str(dict: imageMetadataCustom);
        let imageName = "img.jpg";
        print("button evaluateRDTRestApi 0");
//        currentImageFrame = currentImageFrame.rotate(radians: .pi/2)! // Rotate 90 degrees
//        currentImageFrame = Utils.resizeImage(image: currentImageFrame, newWidth: 1280, newHeight: 720)!
//        print("button evaluateRDTRestApi 1");
        let image = currentImageFrame.jpegData(compressionQuality: 1.0)!;
//        print("button evaluateRDTRestApi 2");
        
        HTTPHandlers.API_POST_FORM_DATA(paramName: paramName, imageData: image, fileName: imageName){(result:String,img:String) in
            print(result);
            callback(result);
        }
        
//        HTTPHandlers.uploadImage(paramName: paramName, fileName: imageName, image: currentImageFrame){ (result) in
//            print(result);
//            callback(result);
//        }
}
}

