import numpy as np
from numpy import linalg as LA
from Utils99 import *
from CLOUD import *
from RLS_VDF import *
from RLS_LOG import *
from RLS_EXP import *
import sys
from sklearn.metrics import mean_squared_error

class TEDARegressor:
  def __init__(self,nI,nR,nO,N1,N2,N3,tau,m,eol,fator=1,ep=0.1,wta=False):

    self.g = 1
    self.gCreated = 1
    self.c= np.array([DataCloud(self.gCreated,nI,nR,nO,N1,N2,N3,tau,x=0)],dtype=DataCloud)
    self.alfa= np.array([0.0],dtype=float)
    self.intersection = np.zeros((1,1),dtype=int)
    self.listIntersection = np.zeros((1),dtype=int)
    self.matrixIntersection = np.zeros((1,1),dtype=int)
    self.relevanceList = np.zeros((1),dtype=int)
    self.k=1
    self.m = m
    self.nI = nI
    self.nR = nR
    self.nO = nO
    self.N1 = N1
    self.N2 = N2
    self.N3 = N3
    self.tau = tau
    self.eol = eol
    self.eolX = 0
    self.fator = fator
    self.classIndex = [[1.0],[1.0]]
    self.argMax = []
    self.NumberOfDataClouds = []
    self.cloud_activation = []
    self.cloud_activation2 = []
    self.HI = np.array([])
    self.DSI = np.array([])
    self.eolDSI = 0
    self.HIp = np.array([])
    self.cycleP=np.array([])
    self.rulL = np.array([])
    self.rulP = np.array([])
    self.rulU = np.array([])
    self.rulR = None
    self.TL = False
    self.ep = ep
    self.order = 3
    self.ff = 1
    self.df =1
    self.dt=1
    self.rls = RLS_LogarithmicRegressor(0.9,10000)
    #self.rls = RLS_ExponentialRegressor(0.9,10000)
    self.vec = np.array([])
    self.win_all = wta

  def mergeClouds(self):
    i=0
    while(i<len(self.listIntersection)-1):
      merge = False
      j=i+1
      while(j<len(self.listIntersection)):
        if(self.listIntersection[i] == 1 and self.listIntersection[j] == 1):
          self.matrixIntersection[i,j] = self.matrixIntersection[i,j] + 1;
        nI = self.c[i].n
        nJ = self.c[j].n
        meanI = self.c[i].mean
        meanJ = self.c[j].mean
        meantI = self.c[i].meant
        meantJ = self.c[j].meant
        varianceI = self.c[i].variance
        varianceJ = self.c[j].variance
        tipicalityI = self.c[i].tipicality
        tipicalityJ = self.c[j].tipicality
        winI=self.c[i].rnn.w_in
        winJ=self.c[j].rnn.w_in
        wrecI=self.c[i].rnn.w_rec
        wrecJ=self.c[j].rnn.w_rec
        woutI=self.c[i].rnn.w_out
        woutJ=self.c[j].rnn.w_out
        hiI=self.c[i].rnn.hI
        hiJ=self.c[j].rnn.hI
        nIntersc = self.matrixIntersection[i,j]    
        idI = self.c[i].ID
        idJ = self.c[j].ID

        # Check if the intersection value is greater than the difference between n and intersection.
        if(nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
            #print('o merge é possível')
            #print(f'fundiu {idI} com {idJ}')
            #print(f'ni: {nI} nj: {nJ}')
            #print('self.matrixIntersection[i,j]  :',self.matrixIntersection[i,j]    )
            merge = True

            # update values for the new cloud
            n = nI + nJ - nIntersc
            mean = ((nI * meanI) + (nJ * meanJ))/(nI + nJ)
            variance = ((nI - 1) * varianceI + (nJ - 1) * varianceJ)/(nI + nJ - 2)
            meant = ((nI * meantI) + (nJ * meantJ))/(nI + nJ)
            #tipicality = sys.float_info.epsilon
            tipicality = ((nI*tipicalityI)+(nJ*tipicalityJ))/(nI + nJ)
            w_in = ((winI*tipicalityI)+(winJ*tipicalityJ))/(tipicalityI+tipicalityJ)
            w_rec = ((wrecI*tipicalityI)+(wrecJ*tipicalityJ))/(tipicalityI+tipicalityJ)
            w_out = ((woutI*tipicalityI)+(woutJ*tipicalityJ))/(tipicalityI+tipicalityJ)
            hI = ((hiI*tipicalityI)+(hiJ*tipicalityJ))/(tipicalityI+tipicalityJ)

            # create and update new data cloud
            self.gCreated+=1
            newCloud = DataCloud(self.gCreated,self.nI,self.nR,self.nO,self.N1,self.N2,self.N3,self.tau,x=0)
            newCloud.updateDataCloud(n,mean,meant,variance,tipicality)

            # update intersection list and data cloud list
            self.listIntersection = np.concatenate((self.listIntersection[0:i], np.array([1]), self.listIntersection[i+1:j], self.listIntersection[j+1:np.size(self.listIntersection)]), axis=None)
            self.c = np.concatenate((self.c[0:i], np.array([newCloud]), self.c[i+1:j], self.c[j+1:np.size(self.c)]), axis=None)
            self.c[-1].rnn.w_in = w_in
            self.c[-1].rnn.w_rec = w_rec
            self.c[-1].rnn.w_out = w_out
            self.c[-1].rnn.hI = hI
            self.c[-1].merge = f'G{self.gCreated}: G{idI}+G{idJ}'
            # update intersection matrix
            M0 = self.matrixIntersection

            # remove rows
            M1 = np.concatenate((M0[0:i, :], np.zeros((1, len(M0))), M0[i+1:j, :], M0[j+1:len(M0), :]))
            # remove columns
            M1 = np.concatenate((M1[:, 0:i], np.zeros((len(M1), 1)), M1[:, i+1:j], M1[:, j+1:len(M0)]), axis=1)
            # calculate new column
            col = (M0[:, i] + M0[:, j]) * (M0[:, i] * M0[:, j] != 0)
            col = np.concatenate((col[0:j], col[j+1:np.size(col)]))
            # calculate new row
            lin = (M0[i, :] + M0[j, :]) * (M0[i, :] * M0[j, :] != 0)
            lin = np.concatenate((lin[0:j], lin[j+1:np.size(lin)]))
            # update column
            M1[:, i] = col
            # update row
            M1[i, :] = lin
            M1[i, i+1:j] = M0[i, i+1:j] + M0[i+1:j, j].T

            self.matrixIntersection = M1
        j += 1
      if (merge):
        i = 0
      else:
        i += 1
  
  def uncertainty(self):
    
    XR = self.rulR
    XP = self.rulP
    for i in range(len(XP)):
        if XP[i] == 0: XP[i]=1e-16
    YR = np.abs(XR-XP)
    for i in range(len(YR)):
        if YR[i] == 0: YR[i]=1e-16

    #rls = RLS_LogarithmicRegressor(ff,dt)
    #rls = RLS_ExponentialRegressor(ff,dt)
    i,YP=0,np.array([])
    if not self.TL:
      for x, y in zip(XP, YR):
        pred = self.rls.predict(x)
        YP = np.append(YP,pred)
        self.rls.update(x, y)
    if self.TL:
      for x, y in zip(XP, YR):
        #print('manteve')
        pred = self.rls.predict(x)
        YP = np.append(YP,pred)

    self.rulU = self.rulP + (self.ep*YP)
    self.rulL = self.rulP - (self.ep*YP)

    for i in range(len(self.rulU)):
       if self.rulL[i] < 0: self.rulL[i] = 0
       if self.rulU[i] < 0: self.rulU[i] = 0
       if self.rulU[i] > np.max(self.rulR): 
          self.rulU[i] = self.rulU[i-1]
          #print('entrou')
       if self.rulL[i] > self.rulP[i]:
          mn = self.rulU[i]
          mx = self.rulL[i]
          self.rulL[i]=mn
          self.rulU[i]=mx

  def predict(self,Xcopy):
    #print('gs:',self.g)
    wSum = sum([cloud.tipicality for cloud in self.c])
    ws = np.array([cloud.tipicality/wSum for cloud in self.c]).reshape(-1,1)
    p = np.array([cloud.rnn.predict(Xcopy) for cloud in self.c])
    #for cloud in self.c:
    #   print(cloud.rnn.ht)
    #for cloud in self.c:
    #   (cloud.rnn.restore())
    #print('p:',p,)
    #print('ws:',ws,p)
    p1 = (p*ws)
    #print('p1:',p,)
    p1 = sum([p1[i][-1] for i in range(len(p))])
    #print('p2:',p,)
    if self.win_all:
       p1 = p[np.argmax(ws)][-1]
    #print(p1)
    return p1
  
  def predict2(self,Xcopy):
    wSum = sum([cloud.tipicality for cloud in self.c])
    ws = np.array([cloud.tipicality/wSum for cloud in self.c]).reshape(-1,1)
    p = np.array([cloud.rnn.predict2(Xcopy) for cloud in self.c])
    p1 = (p*ws)
    p1 = sum([p1[i][-1] for i in range(len(p))])
    if self.win_all:
       p1 = p[np.argmax(ws)][-1]
    return p1
  
  def restore_rnn(self):
     for cloud in self.c:
        cloud.rnn.restore()

  def RUL_single(self,X):
    eP,eL,eU=0,0,0
    pP,pU,pL,rulP,rulU,rulL = [0 for i in range(6)]
    xP,xU_max,xU_min,xL_min,xL_max = [X.copy() for i in range(5)]
    pP = self.predict(xP)*self.fator
    eR = np.abs(self.HI[-1]-pP) 
    self.HIp = np.append(self.HIp,pP)
    vL,vU,vP=[],[],[]

    if pP>0:
      self.rls.update(np.abs(pP), eR)
      eP = self.rls.predict(np.abs(pP))
    self.restore_rnn()
    
    while xP[-1]>self.eol:
      pP = self.predict(xP)*self.fator
      #if self.k ==2: print('pP:',pP)
      xP = np.delete(np.append(xP,pP),0)
      rulP=rulP+1
      if rulP ==160: break
      vP.append(pP)
    self.restore_rnn()

    self.rulP = np.append(self.rulP,rulP)
    self.rulL = np.append(self.rulL,rulL)
    self.rulU = np.append(self.rulU,rulU)

    return
  
  def RUL_uncertainty(self,X):

    eP,eL,eU=0,0,0
    pP,pU,pL,rulP,rulU,rulL = [0 for i in range(6)]
    xP,xU_max,xU_min,xL_min,xL_max = [X.copy() for i in range(5)]
    pP = self.predict(xP)*self.fator
    eR = np.abs(self.HI[-1]-pP) 
    self.HIp = np.append(self.HIp,pP)
    vL,vU,vP=[],[],[]

    if pP>0:
      self.rls.update(np.abs(pP), eR)
      eP = self.rls.predict(np.abs(pP))
    
    self.restore_rnn()
    
    while xP[-1]>self.eol:
      pP = self.predict(xP)*self.fator
      #if self.k ==2: print('pP:',pP)
      xP = np.delete(np.append(xP,pP),0)
      rulP=rulP+1
      if rulP ==160: break
      vP.append(pP)
    self.restore_rnn()

    while xL_min[-1] > self.eol:
      #print(xL_min[:3],xL_max[:3])
      pL1 = self.predict(xL_min)*self.fator
      pL2 = self.predict2(xL_max)*self.fator
      eL1 = abs(self.rls.predict(np.abs(pL1)))
      eL2 = abs(self.rls.predict(np.abs(pL2)))

      #if self.k ==2: print('pL1:',pL1)
      #if self.k ==2: print('pL2:',pL2)
      #if self.k ==2: print('eL1:',eL1)
      #if self.k ==2: print('eL2:',eL2)
      #if self.k ==2: print('xL_min:',xL_min[:3])
      #if self.k ==2: print('xL_max:',xL_max[:3])

      pL_max = max([(pL1-eL1*self.ep),(pL1+eL1*self.ep),(pL2-eL2*self.ep),(pL2+eL2*self.ep)])
      pL_min = min([(pL1-eL1*self.ep),(pL1+eL1*self.ep),(pL2-eL2*self.ep),(pL2+eL2*self.ep)])
      xL_min = np.delete(np.append(xL_min,pL_min),0)  
      xL_max = np.delete(np.append(xL_max,pL_max),0)  
      rulL=rulL+1
      if rulL ==160: break
    #if self.k==50: print('pL:',(pL1-eL1*self.ep),(pL1+eL1*self.ep),(pL2-eL2*self.ep),(pL2+eL2*self.ep))
    #if self.k==50: print('eL:',eL1,eL2)
    
    self.restore_rnn()

    while xU_max[-1] > self.eol:
      
      pU1 = self.predict(xU_min)*self.fator
      pU2 = self.predict2(xU_max)*self.fator
      eU1 = abs(self.rls.predict(np.abs(pU1)))
      eU2 = abs(self.rls.predict(np.abs(pU2)))
      pU_max = max([(pU1-eU1*self.ep),(pU1+eU1*self.ep),(pU2-eU2*self.ep),(pU2+eU2*self.ep)])
      pU_min = min([(pU1-eU1*self.ep),(pU1+eU1*self.ep),(pU2-eU2*self.ep),(pU2+eU2*self.ep)])

      xU_min = np.delete(np.append(xU_min,pU_min),0)  
      xU_max = np.delete(np.append(xU_max,pU_max),0)  
      rulU=rulU+1
      if rulU ==160: break
    self.restore_rnn()

    self.rulP = np.append(self.rulP,rulP)
    self.rulL = np.append(self.rulL,rulL)
    self.rulU = np.append(self.rulU,rulU)


    return 

  def adapt(self,x,y):
     #print('cont rnn:',self.k)
     if self.k >5 and self.HI[-1] < self.eol and self.eolX==0:
        self.eolX=self.cycleP[-1]-1
        self.eolDSI=self.DSI[-1]
     self.HI = np.append(self.HI,y[-1])
     tS = sum([cloud.tipicality for cloud in self.c])
     wS = np.array([cloud.tipicality for cloud in self.c])/tS
     #print(wS)
     #print(wS/tS)
     #print('-----')
     if self.win_all:
      for i,cloud in enumerate(self.c):
          #print('tipicality',self.c[0].tipicality)
          cloud.rnn.adapt(x,y,1)
     if not self.win_all:
       for i,cloud in enumerate(self.c):
          #print('tipicality',self.c[0].tipicality)
          cloud.rnn.adapt(x,y,wS[i])

  def coverage(self):
    end = int(self.eolX-self.nI+1)
    #s = int(len(self.rulR)*0.2)
    s=0
    y_real = self.rulR[s:end]
    y_min = self.rulL[s:end]
    y_max = self.rulU[s:end]
    inclusion_values = [(1 if y_min[i] <= y_real[i] <= y_max[i] else 0) for i in range(len(y_real))]
    cvrg = sum(inclusion_values) / len(y_real)
    return cvrg
  
  def specificity(self):
    end = int(self.eolX-self.nI+1)
    #s = int(len(self.rulR)*0.2)
    s = 0
    x = np.array(self.cycleP)[:].copy()
    yR = np.array(self.rulR)[s:end].copy()
    yL = np.array(self.rulL)[s:end].copy()
    yU = np.array(self.rulU)[s:end].copy()
    diff_sum = np.sum(yU-yL)
    rng = np.max(yR)-np.min(yR)
    sp =  max([0,np.mean(1-((yU-yL)/rng))])
    #if sp < 0:
    #   sp=0
    return sp
  
  def erro(self):
    y_true, y_pred = self.rulR[-1].copy(), self.rulP[-1].copy()
    erro = abs(160 - y_pred)/160
    return erro
  
  def MAPE(self,epsilon=1e-10):
    end = int(self.eolX-self.nI+1)
    start = int((self.cycleP[-1])*0.15)
    #start=0
    y_true, y_pred = self.rulR[start:end].copy(), self.rulP[start:end].copy()
    #print(y_true,y_pred)
    p1,p2,p3=int(len(y_true)*0.25),int(len(y_true)*0.5),int(len(y_true)*0.75)
    x = self.cycleP[start:end].copy()
    erro = y_true - y_pred
    prod = (x **2 * erro)
    #for i in range(len(x)):
    #   if i%5==0:
    #      print('i:',i,'mape:',np.mean(np.abs((prod[:i]) / (y_true[:i] + epsilon))))
    '''print('p1:',np.mean(np.abs((prod[:p1]) / (y_true[:p1] + epsilon))))
    print('p2:',np.mean(np.abs((prod[:p2]) / (y_true[:p2] + epsilon))))
    print('p3:',np.mean(np.abs((prod[:p3]) / (y_true[:p3] + epsilon))))
    print('p4:',np.mean(np.abs((prod) / (y_true+ epsilon))))'''

    return np.mean(np.abs((prod) / (y_true + epsilon)))
  
  def MAPE2(self,epsilon=1e-10):
    y_true, y_pred = self.rulR.copy(), self.rulP.copy()
    x = self.cycleP.copy()
    erro = y_true - y_pred
    prod = (x **2) * erro
    return np.mean(np.abs((prod) / (y_true + epsilon)))
  
  def NDEI(self):
    ndei=np.array([])
    for i in range(1,1+len(self.rulP)):
      rmse=np.sqrt(mean_squared_error(self.rulR[:i], self.rulP[:i]))
      std = (np.std(self.rulP[:i]))
      if std==0: std=1e-12
      ndei=np.append(ndei,rmse/std)
    return ndei
        
  def RMSE(self):
    e = np.where(self.cycleP == self.eolX)[0][0] + 1
    s=-int(len(self.cycleP)*0.8)
    
    rulP = self.rulP[s:e].copy()
    rulR = self.rulR[s:e].copy()
    rmse=np.array([])
    for i in range(1,1+len(rulP)):
      val=np.sqrt(mean_squared_error(rulR[:i],rulP[:i]))
      rmse=np.append(rmse,val)
    Xrmse = self.cycleP[s:e].copy()
    p1,p2,p3,p4=int(len(rmse)*0.25),int(len(rmse)*0.5),int(len(rmse)*0.75),int(len(rmse)+1)
    print('p1:',rmse[p1])
    print('p2:',rmse[p2])
    print('p3:',rmse[p3])
    print('p4',rmse[-1])
    print(s,e,self.cycleP[e])
    print('-------------------')
    return Xrmse,rmse  
  
  def TransferLearning(self):
      self.TL = True
      self.k=1
      self.relevanceList = np.zeros((len(self.relevanceList)),dtype=int)
      self.listIntersection = np.zeros((len(self.listIntersection)),dtype=int)
      self.cloud_activation = []
      self.cloud_activation2 = []
      self.rulL = np.array([])
      self.rulP = np.array([])
      self.rulU = np.array([])
      self.cycleP=np.array([])
      self.rulR = None
      self.eolX = 0
      #for cloud in self.c:
      #   cloud.n = 2
      self.HI = []

  def run(self, x):
    X = x
    self.listIntersection = np.zeros((np.size(self.c)), dtype=int)
    aux = np.array([])

    if self.k == 1 and not self.TL:
        self.c[0] = DataCloud(self.gCreated,self.nI,self.nR,self.nO,self.N1,self.N2,self.N3,self.tau,x)
        self.argMax.append(0)
        self.listIntersection[0] = 1
        #self.cloud_activation.append(1)
        self.cloud_activation.append(self.c[0].ID)
        aux = np.append(aux,1)


    elif self.k == 2 and not self.TL:
        # Add data point to the existing DataCloud.
        self.c[0].addDataCloud(X)
        self.argMax.append(0)
        self.listIntersection[0] = 1
        #self.cloud_activation.append(1)
        self.cloud_activation.append(self.c[0].ID)
        aux = np.append(aux,1)

    elif self.k >= 3 or self.TL:

        i = 0
        createCloud = True
        self.alfa = np.zeros((np.size(self.c)), dtype=float)

        # Iterate over existing DataCloud instances.
        for cloud in self.c:
            n = cloud.n + 1
            mean = ((n-1)/n) * cloud.mean + (1/n) * X
            meant = ((n-1)/n) * cloud.meant + (X.dot(X))/n
            variance=meant-mean.dot(mean)
            eccentricity = ((1/n) + ((mean-X).T.dot(mean-X)) / (n*variance))
            typicality = (1 - eccentricity)+sys.float_info.epsilon
            norm_eccentricity = eccentricity / 2

            #print('ID', cloud.ID,'X', X,'n', n,'variance', variance,'norm_eccentricity', norm_eccentricity,
            #      'threshold', (self.m**2 + 1) / (2*n), )

            if (n - 2)==0:
               norm_typicality = sys.float_info.epsilon
            else:
              norm_typicality = (typicality / (n - 2))
            if (norm_eccentricity <= (self.m**2 + 1) / (2*n)):
                #print('entrou', cloud.ID)
                # If the data point fits inside the DataCloud, update it and set createCloud to False.
                cloud.updateDataCloud(n, mean,meant, variance,typicality)
                self.alfa[i] = norm_typicality
                createCloud = False
                self.listIntersection[i] = 1
                #self.cloud_activation.append(i+1)
                self.cloud_activation.append(cloud.ID)
                aux = np.append(aux,cloud.ID)

            else:
                # If the data point doesn't fit inside the DataCloud, set listIntersection for this index to 0.
                self.alfa[i] = norm_typicality
                self.listIntersection[i] = 0
                cloud.tipicality = typicality
            i += 1
        if (createCloud):
            self.gCreated+=1
            # If none of the existing DataClouds can accommodate the data point, create a new DataCloud instance.
            self.c = np.append(self.c, DataCloud(self.gCreated,self.nI,self.nR,self.nO,self.N1,self.N2,self.N3,self.tau,x))
            #print(len(self.c))
            wSum=sum([cloud.tipicality for cloud in self.c[:-1]])
            wT = np.array([cloud.tipicality/wSum for cloud in self.c[:-1]])
            #print(wT)
            #for cloud in self.c[:-1]:
            #   print('cloud.tipicality:',cloud.tipicality,'wT:',wT,'cloud.rnn.w_in:',cloud.rnn.w_in.shape)
            #w_in = sum([cloud.tipicality*cloud.rnn.w_in/wT[j] for j,cloud in enumerate(self.c[:-1])])
            #w_rec = sum([cloud.tipicality*cloud.rnn.w_rec/wT[j] for j,cloud in enumerate(self.c[:-1])])
            #w_out = sum([cloud.tipicality*cloud.rnn.w_out/wT[j] for j,cloud in enumerate(self.c[:-1])])
            #hI = sum([cloud.tipicality*cloud.rnn.hI/wT[j] for j,cloud in enumerate(self.c[:-1])])
            #hF = sum([cloud.tipicality*cloud.rnn.hF/wT[j] for j,cloud in enumerate(self.c[:-1])])

            w_in = self.c[-2].rnn.w_in
            w_rec = self.c[-2].rnn.w_rec
            w_out = self.c[-2].rnn.w_out
            hI = self.c[-2].rnn.hI
            hF = self.c[-2].rnn.hF

            self.c[-1].rnn.w_in = w_in
            self.c[-1].rnn.w_rec = w_rec
            self.c[-1].rnn.w_out = w_out
            self.c[-1].rnn.hI = hI
            self.c[-1].rnn.hF = hF
            self.g = self.g+1
            #self.cloud_activation.append(self.g)
            self.cloud_activation.append(self.c[-1].ID)

            aux = np.append(aux,self.c[-1].ID)
            self.listIntersection = np.insert(self.listIntersection, i, 1)
            self.matrixIntersection = np.pad(self.matrixIntersection, ((0,1),(0,1)), 'constant', constant_values=(0))
        
        self.relevanceList = self.alfa /np.sum(self.alfa)
        self.argMax.append(np.argmax(self.relevanceList))
        self.classIndex.append(self.alfa)
        self.mergeClouds()
        #print('tipicidades:', self.alfa, 'alfa:',2*np.sum(self.alfa)/(len(self.alfa)))
    self.cloud_activation2.append(aux)
    #if self.k>1: #atenção para a contagem da RUL real

       #print('morreu:',self.k, self.eolX-self.nI+1)
    self.cycleP = np.append(self.cycleP,self.nI+self.k-1)
    self.DSI = np.append(self.DSI,X[-1])
    self.k=self.k+1
    self.rulR = np.flip(self.cycleP)-self.nI
    

