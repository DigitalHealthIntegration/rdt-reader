import matplotlib
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3,2)
fig.suptitle('Learning Curves - Custom model and Transfer learning with Inception V3', size='large')
cols = ['{}'.format(col) for col in ["Custom Model","Transfer learning"]]
rows = ['{}'.format(row) for row in ['Loss', 'Mean squared error', 'Classification accuracy']]
for index in range(3):
    trainingloss=[]
    val_loss = []
    epoch = []
    with open("plotable1") as fin:
        for ind,line in enumerate(fin):
            #if ind<206:
                #epoch.append(ind+1)
                #print(line.split(":"))
                if(index==0):
                    try:
                        #epoch.append(ind+1) #5,10 MSE 1,6 Loss 4,9 Cat acc
                        trainingloss.append(float(line.split(":")[1].split(" ")[1]))
                        val_loss.append(float(line.split(":")[6].split(" ")[1]))
                        epoch.append(ind+1)
                    except:
                        pass
                elif(index==1):
                    try:
                        #epoch.append(ind+1) #5,10 MSE 1,6 Loss 4,9 Cat acc
                        trainingloss.append(float(line.split(":")[5].split(" ")[1]))
                        val_loss.append(float(line.split(":")[10].split(" ")[1]))
                        epoch.append(ind+1)
                    except:
                        pass
                elif(index==2):
                    try:
                        #epoch.append(ind+1) #5,10 MSE 1,6 Loss 4,9 Cat acc
                        trainingloss.append(float(line.split(":")[4].split(" ")[1]))
                        val_loss.append(float(line.split(":")[9].split(" ")[1]))
                        epoch.append(ind+1)
                    except:
                        pass
    
        print (len(epoch),len(trainingloss),len(val_loss))
        
        
        if(index==0):
            ax[index,0].plot(epoch, trainingloss, label='train_loss')
            ax[index,0].plot(epoch,val_loss,label='val_loss')
        elif(index==1):
            ax[index,0].plot(epoch, trainingloss, label='train_error')
            ax[index,0].plot(epoch,val_loss,label='val_error')
        elif(index==2):
            ax[index,0].plot(epoch, trainingloss, label='train_acc')
            ax[index,0].plot(epoch,val_loss,label='val_acc')
        ax[index,0].legend()

    trainingloss=[]
    val_loss = []
    epoch = []
    with open("plotable2") as fin:
        for ind,line in enumerate(fin):
            if ind<206:
                #epoch.append(ind+1)
                #print(line.split(":"))
                if(index==0):
                    try:
                        #epoch.append(ind+1) #5,10 MSE 1,6 Loss 4,9 Cat acc
                        trainingloss.append(float(line.split(":")[1].split(" ")[1]))
                        val_loss.append(float(line.split(":")[6].split(" ")[1]))
                        epoch.append(ind+1)
                    except:
                        pass
                elif(index==1):
                    try:
                        #epoch.append(ind+1) #5,10 MSE 1,6 Loss 4,9 Cat acc
                        trainingloss.append(float(line.split(":")[5].split(" ")[1]))
                        val_loss.append(float(line.split(":")[10].split(" ")[1]))
                        epoch.append(ind+1)
                    except:
                        pass
                elif(index==2):
                    try:
                        #epoch.append(ind+1) #5,10 MSE 1,6 Loss 4,9 Cat acc
                        trainingloss.append(float(line.split(":")[4].split(" ")[1]))
                        val_loss.append(float(line.split(":")[9].split(" ")[1]))
                        epoch.append(ind+1)
                    except:
                        pass        
        
        if(index==0):
            ax[index,1].plot(epoch, trainingloss, label='train_loss')
            ax[index,1].plot(epoch,val_loss,label='val_loss')
        elif(index==1):
            ax[index,1].plot(epoch, trainingloss, label='train_error')
            ax[index,1].plot(epoch,val_loss,label='val_error')
        elif(index==2):
            ax[index,1].plot(epoch, trainingloss, label='train_acc')
            ax[index,1].plot(epoch,val_loss,label='val_acc')
        ax[index,1].legend()

for axx, col in zip(ax[0], cols):
    axx.set_title(col, size='large')

for axx, row in zip(ax[:,0], rows):
    axx.set_ylabel(row, rotation=90)

fig.tight_layout()
fig.set_size_inches(10.5, 18.5)

fig.savefig("Learning_curve.jpg")
