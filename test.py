import flasker
import cv2
import glob



servObj = flasker.FluServer()
root_dir = "D:\\source\\repos\\audere\\new_images_sarvesh\\"
y_truth =[]
y_pred = []
failed_images = []
with open("images_used_in_training_redblue.txt") as fin:
    fileUsedIntrainingRB=fin.readlines()
    fileUsedIntrainingRB=[root_dir+x.strip().split()[1] for x in fileUsedIntrainingRB]
print(fileUsedIntrainingRB)
with open("out_old_model.csv","w") as fout:
    fout.write("Truth,predicted,filename\n")
    for filename in glob.iglob(root_dir + '**/*.jpg', recursive=True):
        img = cv2.imread(filename)
        tmp_pred=flasker.runPipeline(img,servObj)
        y_pred.append(tmp_pred)
        
        if "Negative" in filename:
            y_truth.append(0)
            if tmp_pred != 0:
                failed_images.append(filename)
        elif "FluA+B" in filename:
            y_truth.append(3)
            if tmp_pred != 3:
                failed_images.append(filename)

        elif "FluA" in filename:
            y_truth.append(1)
            if tmp_pred != 1:
                failed_images.append(filename)

        elif "FluB" in filename:
            y_truth.append(2)
            if tmp_pred != 2:
                failed_images.append(filename)
        fout.write(str(y_truth[-1])+","+str(tmp_pred)+","+filename.replace("/","\\")+"\n")
        # break
        print(filename)
        