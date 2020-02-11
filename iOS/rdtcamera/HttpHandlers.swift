//
//  HttpHandlers.swift
//  RDT Camera
//
//  Created by developer on 24/01/20.
//  Copyright © 2020 IPRD. All rights reserved.
//

import UIKit

public class HTTPHandlers {
    

    static func API_POST_FORM_DATA(paramName:String, imageData:Data, fileName:String, callback: @escaping (String,String) -> ())
    {

        let API_URL = "http://100.24.50.45:9000/Quidel/QuickVue"
        print("API_URL : \(API_URL)")
        let request = NSMutableURLRequest(url: URL(string: API_URL)!)
        request.httpMethod = "POST"

        let boundary = generateBoundaryString()

        //define the multipart request type

        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")


        let body = NSMutableData()

        let fname = fileName
        let mimetype = "image/jpeg"

        //define the data post parameter

        body.append("--\(boundary)\r\n".data(using: String.Encoding.utf8)!)

        body.append("Content-Type: \(mimetype)\r\n\r\n".data(using: String.Encoding.utf8)!)

        body.append("Content-Disposition:form-data; name=\"image\"; filename=\"\(fname)\"\r\n".data(using: String.Encoding.utf8)!)
        body.append(imageData)
        
        
        body.append("\r\n".data(using: String.Encoding.utf8)!)

        body.append("--\(boundary)--\r\n".data(using: String.Encoding.utf8)!)

        body.append("--\(boundary)\r\n".data(using: String.Encoding.utf8)!)
        let key: String = "metadata";
        body.append("Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n".data(using: String.Encoding.utf8)!)
//        body.append("\(paramName)\r\n".data(using: String.Encoding.utf8)!)
        body.append("\(paramName)".data(using: String.Encoding.utf8)!)
        
        
        let imageKey = "image";
        
        body.append("\r\n--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"\(imageKey)\"; filename=\"\(fileName)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)

        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        

        request.httpBody = body as Data

        // return body as Data
        print("Fire....")
        let session = URLSession.shared
        let task = session.dataTask(with: request as URLRequest) {
            (
            data, response, error) in
            print("Complete")
//            print("data", data)
            let stringResponse = String(data: data!, encoding: .utf8)
//            print("data2", stringResponse)
            
//            let x = stringResponse?.split(separator: Character.init(boundary));
            let x = stringResponse?.components(separatedBy: boundary)
            
//            let boundryStart = stringResponse!.index(of: "\n")
//            let boundryEnd = stringResponse!.index(of: "\nContent-Disposition")
            
            let requestBoundry = stringResponse!.substring(with: 1..<37);
            //print("requestBoundry", requestBoundry);
            let splits = stringResponse?.components(separatedBy: requestBoundry);
            //print("splits", splits);
            
            let valueStart = splits![1].index(of: "{")
            
            var valueEnd = splits![1].index(of: "\n\n")
            if (valueEnd==nil){
                valueEnd = splits![1].index(of: "\n-")
            }
            var responseValues = "{\"msg\":\"error\"}"
            do{
                
                responseValues  = splits![1].substring(with: valueStart!..<valueEnd!)

                
            }catch let error as NSError {
                print("Failed to load: \(error.localizedDescription)")
            }
            //print("responseValues", responseValues)
            var base64Values = ""

            if splits?.count==3{
                let base64Start = splits![2].index(of: "/9j/");
                let base64End = splits![2].index(of: "\n--");
                
                base64Values = splits![2].substring(with: base64Start!..<base64End!)
                
            }else{
                base64Values = ""
            }
            
            
                //print("base64Values", base64Values)
            
            
            //print("data 3", x);
            callback(responseValues,base64Values);
            // add some validation when something fails TODO
            return;
//            let imageData = NSData(base64EncodedString: data?.base64EncodedString(), options: .allZeros)
//            imageData.get
//            print("response", response)
//            print("error", error)
            if error != nil
            {
                print("error upload : \(error)")
                // callback here
                callback(responseValues,base64Values);
                return
            }else{
                callback(responseValues,base64Values);
            }

//            do
//            {
//
//                if let json = try JSONSerialization.jsonObject(with: data!, options: .allowFragments) as? [String: Any]
//                {
//                    callback(json as NSDictionary?)
//                }else
//                {
//                    print("Invalid Json")
//                }
//            }
//            catch
//            {
//                print("Some Error")
////                callback(nil)
//            }
        }
        task.resume()
    }

    static func generateBoundaryString() -> String {
        return "Boundary-\(NSUUID().uuidString)"
    }
    
    
    
    public static func hitAPI(paramName: String, fileName: String, image: UIImage, callback: @escaping (Any,Any) -> ()) -> () {
       let configuration = URLSessionConfiguration.default
       let session = URLSession(configuration: configuration)
       let url = URL(string: "http://100.24.50.45:9000/Quidel/QuickVue")
       var request : URLRequest = URLRequest(url: url!)
       request.httpMethod = "POST"
       request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
       request.addValue("application/json", forHTTPHeaderField: "Accept")
        
       let dataTask = session.dataTask(with: url!) { data,response,error in
          // 1: Check HTTP Response for successful GET request
          guard let httpResponse = response as? HTTPURLResponse, let receivedData = data
          else {
             print("error: not a valid http response")
             return
          }
          switch (httpResponse.statusCode) {
             case 200:
                //success response.
                break
             case 400:
                break
             default:
                break
          }
       }
       dataTask.resume()
    }
    
    public static func uploadImage(paramName: String, fileName: String, image: UIImage, callback: @escaping (Any,Any) -> ()) -> () {
        let url = URL(string: "http://100.24.50.45:9000/Quidel/QuickVue")

        // generate boundary string using a unique per-app string
        let boundary = UUID().uuidString

        let session = URLSession.shared

        // Set the URLRequest to POST and to the specified URL
        var urlRequest = URLRequest(url: url!)
        urlRequest.httpMethod = "POST"

        // Set Content-Type Header to multipart/form-data, this is equivalent to submitting form data with file upload in a web browser
        // And the boundary is also set here
        urlRequest.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var data = Data()
        
        data.append("\r\n--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; metadata=\"\(paramName)\"; filename=\"\(fileName)\"\r\n".data(using: .utf8)!)
        data.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
        
        
        // default behavior
//        data.append("\r\n--\(boundary)\r\n".data(using: .utf8)!)
//        data.append("Content-Disposition: form-data; name=\"\(paramName)\"; filename=\"\(fileName)\"\r\n".data(using: .utf8)!)
//        data.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
//        data.append(image.jpegData(compressionQuality: 1.0)!)
//
//        data.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)



//        // Add the image data to the raw http request data
//        data.append("\r\n--\(boundary)\r\n".data(using: .utf8)!)
//        data.append("Content-Disposition: form-data; metadata=\"\(paramName)\"; image=\"\(image)\"\r\n".data(using: .utf8)!)
////        data.append("Content-Disposition: form-data; metadata=\"\(paramName)\"\r\n".data(using: .utf8)!)
////        data.append("Content-Disposition: form-data; image=".data(using: .utf8)!)
//        //data.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
//        //data.append(image)
//        print("image base64", image);
//        data.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        
        
        
        
        
        // urlRequest.setValue("\(data.count)", forHTTPHeaderField:"Content-Length")
        print("button api 1");
        print(data);
        print("button api 2");
        print(urlRequest.allHTTPHeaderFields);
        // Send a POST request to the URL, with the data we created earlier
        session.uploadTask(with: urlRequest, from: data, completionHandler: { responseData, response, error in
            if error == nil {
                print("button api 2");
                print(responseData);
                print(response)
                let jsonData = try? JSONSerialization.jsonObject(with: responseData!, options: .allowFragments)
                print("button api 3");
                if let json = jsonData as? [String: Any] {
                    print(json)
                    callback(json,"");
                    print("button api 4");
                }
                print("button api 5");
                
            }
        }).resume()
    }
}
extension String {
    func index(from: Int) -> Index {
        return self.index(startIndex, offsetBy: from)
    }

    func substring(from: Int) -> String {
        let fromIndex = index(from: from)
        return String(self[fromIndex...])
    }

    func substring(to: Int) -> String {
        let toIndex = index(from: to)
        return String(self[..<toIndex])
    }

    func substring(with r: Range<Int>) -> String {
        let startIndex = index(from: r.lowerBound)
        let endIndex = index(from: r.upperBound)
        return String(self[startIndex..<endIndex])
    }
}

extension StringProtocol {
    func index<S: StringProtocol>(of string: S, options: String.CompareOptions = []) -> Index? {
        range(of: string, options: options)?.lowerBound
    }
    func endIndex<S: StringProtocol>(of string: S, options: String.CompareOptions = []) -> Index? {
        range(of: string, options: options)?.upperBound
    }
    func indices<S: StringProtocol>(of string: S, options: String.CompareOptions = []) -> [Index] {
        var indices: [Index] = []
        var startIndex = self.startIndex
        while startIndex < endIndex,
            let range = self[startIndex...]
                .range(of: string, options: options) {
                indices.append(range.lowerBound)
                startIndex = range.lowerBound < range.upperBound ? range.upperBound :
                    index(range.lowerBound, offsetBy: 1, limitedBy: endIndex) ?? endIndex
        }
        return indices
    }
    func ranges<S: StringProtocol>(of string: S, options: String.CompareOptions = []) -> [Range<Index>] {
        var result: [Range<Index>] = []
        var startIndex = self.startIndex
        while startIndex < endIndex,
            let range = self[startIndex...]
                .range(of: string, options: options) {
                result.append(range)
                startIndex = range.lowerBound < range.upperBound ? range.upperBound :
                    index(range.lowerBound, offsetBy: 1, limitedBy: endIndex) ?? endIndex
        }
        return result
    }
}
