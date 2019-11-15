const async = require('async');
const fs = require('fs');
const path = require('path');
const https = require('request');
/*var csv = require('fast-csv');
var ws = fs.createWriteStream('my1.csv');*/
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
var cn = 0;
const csvWriter = createCsvWriter({
  path: 'rdtTestResult.csv',
  header: [
		{id: 'fileName', title: 'File Name'},
	  {id: 'flutype', title: 'Flutype'},	
    {id: 'result', title: 'Result'},
    {id: 'rc', title: 'RC'},
    {id: 'msg', title: 'Message'},
    {id: 'includeProof', title: 'Include Proof'},
    {id: 'errorMsg', title: 'Error Message'}
  ]
});



const folderPath = 'C:/ws/MGD/RDT/rdt-reader/RDT_Images-20191010T070521Z';//'/home/developer/Documents/RDT Test Image';//RDT Test Image';//RDTLess
var arr1 = [];
var flist = [];
console.log("Please wait!! \n",folderPath);

var searchRecursive = function(dir, pattern) {
	// This is where we store pattern matches of all files inside the directory
	var results = [];

	// Read contents of directory
	fs.readdirSync(dir).forEach(function (dirInner) {
		// Obtain absolute path
		dirInner = path.resolve(dir, dirInner);

		// Get stats to determine if path is a directory or a file
		var stat = fs.statSync(dirInner);

		// If path is a directory, scan it and combine results
		if (stat.isDirectory()) {
			results = results.concat(searchRecursive(dirInner, pattern));
		}

		// If path is a file and ends with pattern then push it onto results
		if (stat.isFile() && dirInner.endsWith(pattern)) {
			results.push(dirInner);
		}
	});

	return results;
};

flist = searchRecursive(folderPath, '.jpg'); // replace dir and pattern
																									// as you seem fit	
console.log("FileList !! \n",flist);

var c=0;
async.eachSeries(flist,function(file,callback){
	startTesting(file,function(err,data){
		if(err){
			console.log(JSON.stringify(err));
		}
		callback();
	});
},function(err){
	console.log(err?JSON.stringify(err):"Done calling")
});
/* for(i=0;i<flist.length;i++){
	startTesting(flist[i]);
} */
//##########################################################################################################
function startTesting(fname,cb){ 
	async function requestAPI(fname){
		{

			  var promise = new Promise(function(resolve, reject) {
			    try {
						delfiles('roi.jpg');
						delfiles('out.jpg');
						delfiles('yolopred.jpg');
						delfiles('rdt_crop.jpg');
						delfiles('resized.jpg');
						delfiles('rotated.jpg');
						delfiles('translated.jpg');
						var dataToBeSend = {};
						dataToBeSend.metadata = '{"UUID":"a432f9681-a7ff-43f8-a1a6-f777e9362654","Quality_parameters":{"brightness":"10"},"RDT_Type":"Flu_Audere","Include_Proof":"True"}';
						dataToBeSend.image = fs.createReadStream(fname);
						const url = 'http://127.0.0.1:9000/Quidel/QuickVue';		            
						console.log('Sending file '+ fname);
						https.post({url:url, formData: dataToBeSend}, function optionalCallback(err, httpResponse, body) {
						resolve(body);
						});		            
			    }
			    catch(error) {
			        reject(error);
			    }
			});

			  promise
				.then(function(value) {
				  console.log(c,fname,'>> RDT IMAGE TEST > ',++cn);	
					var testData = value.substring(value.indexOf('{'), value.indexOf('}')+1);
					resultToBeDisplay(fname, checkTheTest(value.toString(),fname,testData));
					saveImageJpeg(value,fname);
			    console.log('==================================================================================================================');
				    /* setTimeout(() => {
					     requestAPI(++c,fileList);
						}, 100); */
						cb();
				}, function(error) {
					cb(error)
				    // throw error;  // rethrow the error
				});
		}
	};
function copyfiles(dir,fname){
	console.log('copy -> '+ fname +'->'+ dir+'/'+fname)
	if(fs.existsSync(fname)){
		fs.copyFileSync(fname, dir+'/'+fname, (err) => {
			if (err) throw err;
			console.log(fname + ' was copied to '+ dir);
		});
	}
};
function delfiles(fname){
	console.log('delete -> '+ fname)
	var fs = require('fs');
	if(fs.existsSync(fname))
		fs.unlinkSync(fname);
};

function saveImageJpeg(value,fileName){
	var str1 = value.substring(value.indexOf('image/jpeg\r\n\r\n')+10);
	var imgBase64String = str1.substring(0,str1.indexOf('--'));
	var fs = require('fs');
	var dir1 = './output';
	if (!fs.existsSync(dir1)){
			fs.mkdirSync(dir1);
	}
	var fname= fileName.substring(fileName.lastIndexOf("\\")+1, fileName.length-4);
	var dir2 = dir1+'/'+fname;

	if (!fs.existsSync(dir2)){
			fs.mkdirSync(dir2);
	}
	console.log('Copying file '+ fileName);
	copyfiles(dir2,'out.jpg');
	copyfiles(dir2,'yolopred.jpg');
	copyfiles(dir2,'rdt_crop.jpg');
	copyfiles(dir2,'resized.jpg');
	copyfiles(dir2,'roi.jpg');
	copyfiles(dir2,'rotated.jpg');
	copyfiles(dir2,'translated.jpg');
	require("fs").writeFile(dir2+'/'+"rxed_crop.jpg", imgBase64String, 'base64', function(err) {
	//console.log(err);

	});
	};

function readFiles(dirname, onFileContent, onError) {
  let fileList = [];
  fs.readdir(dirname, function(err, filenames) {
	    if (err) {
	      //onError(err);
	      console.log("Error : (please check your file path)");
	      console.log("e.g, /home/developer/rdt test image/Btype \n     /home/developer/Downloads/Negative-20190927T075516Z-001/Negative/Morning");
	      console.log();
	      return ;
	    }
	    filenames.forEach(function(filename) {
	      fileList.push(filename);
	      fs.readFile(dirname + filename, 'utf-8', function(err, content) {
	        if (err) {
	          //onError(err);
	          return;
	        }
	        onFileContent(filename, content);	        
	      });
	    });
	  
	  requestAPI(0,fileList)
	  });  
}

function checkTheTest(responseBody,filePath, testData){
		console.log(testData);
		let returnData = {};
		var testName = filePath;
		let toBeMatchWith = "";
		if(filePath.indexOf("FluA+B")>= 0){
			toBeMatchWith = '{"UUID": "a432f9681-a7ff-43f8-a1a6-f777e9362654", "rc": "3", "msg": "A+Btype", "Include Proof": "True"}';
		}else if(filePath.indexOf("FluB")>= 0){
			toBeMatchWith = '{"UUID": "a432f9681-a7ff-43f8-a1a6-f777e9362654", "rc": "2", "msg": "Btype", "Include Proof": "True"}';
		}else if(filePath.indexOf("FluA")>= 0){
			toBeMatchWith = '{"UUID": "a432f9681-a7ff-43f8-a1a6-f777e9362654", "rc": "1", "msg": "Atype", "Include Proof": "True"}';
		}else if(filePath.indexOf("Negative")>= 0){
			toBeMatchWith = '{"UUID": "a432f9681-a7ff-43f8-a1a6-f777e9362654", "rc": "0", "msg": "No Flu", "Include Proof": "True"}';
		}else {
			console.log("=> ",testName);
		}

			if(testData.toString().trim() == toBeMatchWith.trim()){
				returnData.result = "PASS";
				returnData.actual = testData+"";
				returnData.expected = toBeMatchWith+"";
				return returnData;
			}else{
				//return "FAIL \n\n\n EXPECTED DATA "+toBeMatchWith+"\n ACTUAL DATA  "+responseBody;
				returnData.result = "FAIL";
				returnData.actual = testData+"";
				returnData.expected = toBeMatchWith+"";
				return returnData;
			}
		}

	function resultToBeDisplay( fileName,testData){
		//console.log(fileName,"=>",testData.actual);
		console.log();	
		var resObj = {};
		var expObj = {};
		
		try {
		   resObj = JSON.parse(testData.actual);
		  } catch (e) {
		    resObj.rc = "-";
		    resObj.msg = "-";
		    resObj.includeProof = "";
		    resObj.errorMsg = JSON.stringify(testData);
		  }

			try {
				expObj = JSON.parse(testData.expected);
			 } catch (e) {
				expObj.rc = "-";
				expObj.msg = "-";
				expObj.includeProof = "";
				expObj.errorMsg = JSON.stringify(testData);
			 }
 
		const data = [{
			
			fileName : fileName,
			flutype : expObj['msg'],
			result : testData.result,
			rc : resObj.rc,
			msg : resObj.msg,
			includeProof : resObj['Include Proof'],
			errorMsg : resObj.errorMsg,
		}];

		csvWriter
		  .writeRecords(data)
		  //.then();  //()=> console.log("#")

	}
	requestAPI(fname);
}