import numpy as np
import sys
import random
from copy import copy
from graphviz import Digraph
dot = Digraph(comment='The Round Table')
class Binary_Decision_Tree:
    
    def __init__(self):
        self.training_data=np.array([])
        self.labels=[]
        self.labelsT=[]
        self.genes=[]
        self.node_num=0
        self.test_data=np.array([])
        self.already_used_attributes=[]
        self.Tree=np.ones((10000,1))
        self.Thresholds=np.ones((10000,1))
        self.decisions={} 
	self.Tree=-1*self.Tree
	self.attributes_selected=[]
        pass
    
    def calculate_IG(self,data1,threshold,Attr,labels1):
        data=copy(data1)
        labels=copy(labels1)
        
        Left_rows=np.where(data[:,Attr]>=threshold)[0]
        Right_rows=np.where(data[:,Attr]<threshold)[0]
        
        #Calculate parent threshold 
        Rows_healthy=np.where(labels==1)[0]
        Rows_trisomy=np.where(labels==0)[0]
        if(labels.shape[0] ==0):
            HParent=0
	else: 
		pH=float(Rows_healthy.shape[0])/labels.shape[0]
                pC=float(Rows_trisomy.shape[0])/labels.shape[0]
        	
        	if pH==0 or pC==0:
        	    HParent=0
        	else:
        	    HParent=-1*pH*np.log2(pH) - pC*np.log2(pC)
        
        labelsLeft=copy(labels[Left_rows])
        labelsRight=copy(labels[Right_rows])
        # calculate the threshold For Left Child
        Rows_healthy=np.where(labelsLeft==1)[0]
        Rows_trisomy=np.where(labelsLeft==0)[0]
        
        if(labelsLeft.shape[0] ==0 ):
            Hleft_child=0
	else:
        	Prob_HL=float(Rows_healthy.shape[0])/labelsLeft.shape[0]
        	Prob_TL=float(Rows_trisomy.shape[0])/labelsLeft.shape[0]
        	if Prob_HL==0 or Prob_TL==0:
        	    Hleft_child=0
        	else:
        	    Hleft_child=-1*Prob_HL*np.log2(Prob_HL) - Prob_TL*np.log2(Prob_TL)
        	    Hleft_child=Hleft_child*float(Left_rows.shape[0])/data.shape[0]
        
        
        # calculate the threshold For Right Child
        Rows_healthy=np.where(labelsRight==1)[0]
        Rows_trisomy=np.where(labelsRight==0)[0]
        
        if(labelsRight.shape[0] ==0 ):
            Hright_child=0
        
	else:
        	Prob_HR=float(Rows_healthy.shape[0])/labelsRight.shape[0]
        	Prob_TR=float(Rows_trisomy.shape[0])/labelsRight.shape[0]
        	if Prob_HR==0 or Prob_TR==0:
        	    Hright_child=0
        	else:
        	    Hright_child=-1*Prob_HR*np.log2(Prob_HR) - Prob_TR*np.log2(Prob_TR)
        	    Hright_child=Hright_child*float(Right_rows.shape[0])/data.shape[0]
        
        IG=HParent-Hleft_child-Hright_child
        return IG        
        
        
       
    def findThresholdAndIG(self,data1,Attr,labels1):
        data=copy(data1)
        labels=copy(labels1)
        values=data[:,Attr]
	classes= copy(labels1)
        values=list(values)
	classes= list(classes)
	values = values,classes
	values = np.array(values)
	values=values.transpose()
	values = values[values[:,0].argsort()]
	values=values.transpose()
        toTryThreshholds=[]
        for i in range(0,len(values[0])-1):
	   if(values[1][i] != values[1][i+1]):
           	toTryThreshholds.append((values[0][i]+values[0][i+1])/2)

        IG=[]
        #calculate the Informatin gain for all thresholds
        for threshold in toTryThreshholds:
            IG.append(self.calculate_IG(data,threshold,Attr,labels))
        
        #select one with the maximum infomation gain
        maxIG=max(IG)
        Thresh_max=IG.index(maxIG)
        
        return toTryThreshholds[Thresh_max],maxIG
            
        
                                
        
    def construct_tree(self,data1,nodeNum,labels1):
            #since its a recursive function we need to have a base case. return when the number of wrong classes is 0. maybe 
            #we can chane it later
            print 'nodeNum is ',nodeNum
            data=copy(data1)
            labels=copy(labels1)
            rows=np.where(labels==1)[0] # separate the rows with healthy and trisomic
            rows2=np.where(labels==0)[0]

            # if leaf node create a decision dictonary and a graphviz leafnode
            if rows.shape[0]==0 or rows2.shape[0]==0 or nodeNum >=32: # Restrict the number of levels to 5
                self.decisions[nodeNum]=(rows.shape[0],rows2.shape[0])
		if(rows.shape[0] == 0) :
            	       node_data = '\nsamples=%s,\nSpam=%s,  Not spam=%s \n class = Not Spam'%(data.shape[0],rows.shape[0],rows2.shape[0])
                       dot.node("leaf"+str(nodeNum), label = node_data)
                       return 'leaf'+str(nodeNum)
		else:
            	       node_data = '\nsamples=%s,\nSpam=%s,  Not spam=%s \n class = Spam'%(data.shape[0],rows.shape[0],rows2.shape[0])
                       dot.node("leaf"+str(nodeNum), label = node_data)
                       return 'leaf'+str(nodeNum)

           #if interneal node 
            IGA=[]
            thresholds=[] 
	   # check the  information gain of all the unused attributes
	    for attr in range(0,len(data[0])):
		if(attr not in self.already_used_attributes):
			thresh,IG=self.findThresholdAndIG(data,attr,labels)
			IGA.append(IG)
			thresholds.append(thresh)
		else:
			IGA.append(0)
			thresholds.append(0)
			
            #select one with the maximum information gain 
            maxIG=max(IGA)
            Attr=IGA.index(maxIG)
            thresh=thresholds[Attr]
	    self.attributes_selected.append(Attr)
            #create the graphviz internal node
            node_data = '\nsamples=%s,\nSpam=%s,  Not spam=%s'%(data.shape[0],rows.shape[0],rows2.shape[0])
	    if(rows.shape[0] > rows2.shape[0]):
		node_data = node_data+"\nClass = Spam"
            else:
		node_data = node_data+"\nClass = Not Spam"
            dot.node(str(Attr), label = str(Attr)+'\n<='+(str(thresh))+node_data)
            self.already_used_attributes.append(Attr)
            self.Tree[nodeNum]=Attr
            self.Thresholds[nodeNum]=thresh
            rows=np.where(data[:,Attr]>=thresh)[0]
            rows2=np.where(data[:,Attr]<thresh)[0]
            
            dataLeft=copy(data[rows])
            dataRight=copy(data[rows2])
            
            labelsLeft=copy(labels[rows])
            labelsRight=copy(labels[rows2])
            print '\n\n'
            # call recursive constuciton of the tree
            res1 = self.construct_tree(dataLeft,2*nodeNum,labelsLeft)
	    dot.edge(str(Attr),res1,label='False')             # create an edge between parent and leftchild
            res2 = self.construct_tree(dataRight,2*nodeNum+1,labelsRight)
	    dot.edge(str(Attr),res2,label='True')              # create an edge between parent and leftchild
	    return str(Attr)

            
            
            
            
                
        
    
    def load_train_data(self):
        
        f=open(sys.argv[1]) 
#	row=f.readline()
#        row=row.rstrip()   
#        self.genes=row.split(',') #get the gene names separately
#        
        for row in f:        #Read input data as rows
           
            self.AllValues={}
            
            row=row.rstrip()   
            attributes=row.split(',')
            attributes_2=[float(i) for i in attributes[0:len(attributes)-1]]
            self.labels.append(float(attributes[-1]))
            
            attributes=copy(np.asarray(attributes_2))
            attributes=attributes.reshape(1,len(attributes))
            if self.training_data.shape[0]==0:
                self.training_data=copy(attributes)
            else:
                self.training_data=copy(np.vstack((self.training_data,attributes)))
                
        self.labels=copy(np.asarray(self.labels))
        self.labels=copy(self.labels.reshape(-1,1))
        
		

	print self.labels.shape[0]
        print 'Constructing the tree'
        
        self.construct_tree(self.training_data,1,self.labels)
        #test the training data
        self.test(self.training_data,self.labels)
    
    def load_test_data(self):
	print "Decision tree is",self.decisions
	print "\n \n"
        
        f=open(sys.argv[2])
#	row=f.readline()
#        row=row.rstrip()   
#        classes=row.split(',')
#        
        for row in f:
           

            
            row=row.rstrip()   
            row=row[0:len(row)-1]
            attributes=row.split(',')
            attributes_2=[float(i) for i in attributes[0:len(attributes)-1]]
            self.labelsT.append(float(attributes[-1]))
            
            attributes=copy(np.asarray(attributes_2))
            attributes=attributes.reshape(1,len(attributes_2))
            if self.test_data.shape[0]==0:
                self.test_data=copy(attributes)
            else:
                self.test_data=copy(np.vstack((self.test_data,attributes)))
                
        self.labelsT=copy(np.asarray(self.labelsT))
        self.labelsT=copy(self.labelsT.reshape(-1,1))
	
           
        self.test(self.test_data,self.labelsT)
    
    
    def calculate_accuracy(self,gold,predicted):
        output=copy(gold.tolist())
        predic=copy(predicted.tolist())
        
        correct=0
        for i in range(0,len(output)):
            if output[i][0]==predic[i]:
                correct+=1
        
        return 100*(float(correct)/len(output))
        
    def find_edge(self,data1,nodeNum):
        data=copy(data1)
        #print 'nodeNum is ',nodeNum
	#if the node is a leaf node the we check the decisions dictionary for result
        if self.Tree[nodeNum][0]==-1:
            #then we check the decisions 
	
            healthy,colic=self.decisions[nodeNum]
            if healthy>0:
                res= 1
            else:
                res= 0

	#if the node is an inter node then we travers left or right of tree 
        elif data[self.Tree[nodeNum][0]]>=self.Thresholds[nodeNum][0]:
            #go left
            res=self.find_edge(data,2*nodeNum)
        else:
            #go right
            res=self.find_edge(data,2*nodeNum+1)
           
        return res
                    
    def test(self,data1,labels1):
        data=copy(data1)
        labels=copy(labels1)
        predicted=[]
        # Test for each row and predict output
        for i in range(0,data.shape[0]):
            
            res=self.find_edge(data[i],1)
            predicted.append(res)
            print 'testing ',i,' predicted= ',res,' output is ',labels[i][0]
            
            
        predicted=np.asarray(predicted)
        accuracy=self.calculate_accuracy(labels,predicted)
        print 'Accuracy is ',accuracy,'%'
	f = open ("selected_attributes",'w')
	f.write(str(self.attributes_selected))
        
if(len(sys.argv)!=3):
	print "Usage dt.py filename1 filename2"
else:
	
	OBJECT=Binary_Decision_Tree()
	OBJECT.load_train_data()
	OBJECT.load_test_data()
a = open ("spam.dot",'w')
a.write(dot.source)

