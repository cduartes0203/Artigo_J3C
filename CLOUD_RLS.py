import numpy as np
from RTLO import *
from RLS_VDF import *

class DataCloud:
  N=0
  def __init__(self,ID,nF,nS,FF,DF,DT,x):
    
      self.ID = ID
      self.n=1
      self.mean=x
      self.variance=0
      self.meant = np.array(x).dot(np.array(x))
      self.pertinency=1
      #self.tipicality=sys.float_info.epsilon
      self.tipicality=1e-12
      self.nS = nS
      self.nF = nF
      self.FF = FF
      self.DF = DF
      self.DT = DT
      self.rls = RLS_VDF(nS,nF,FF,DF,DT)
      self.merge = None
      DataCloud.N+=1

  def show(self):
        names = ['n','mean','meant','tipicality','variance','w']
        for key, value in self.__dict__.items():
            if key in names:
                print(f"{key}: {value}")
        print('rls.w: ',self.rnn.w_in)
      
  def addDataCloud(self,x):
      self.n=2
      self.mean=(self.mean+x)/2
      self.meant=((self.meant)/2) + (x.dot(x))/2
      self.variance=self.meant-self.mean.dot(self.mean)
      
  def updateDataCloud(self,n,mean,meant,variance,tipicality):
      self.n=n
      self.mean=mean
      self.meant=meant
      self.variance=variance
      self.tipicality=tipicality