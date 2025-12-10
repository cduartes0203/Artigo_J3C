import numpy as np

class RLS_VDF():
    def __init__(self,ns,nf,ff,df,dt):
        nf=nf+1
        self.parameters = nf
        self.w_0 = np.zeros(nf)
        self.w =  self.w_0
        self.lamda = ff
        self.e = df
        self.P = dt*np.eye(self.parameters)
        self.A = np.linalg.inv(np.eye(self.parameters,self.parameters))
        self.n_samples = ns
        
    def adapt(self,phi,y,wS=1):
        #phi = self.Afine(phi)
        #print('phi:',phi)
        for i in range(self.n_samples-1):
            print('-------------------------------------------')
            U, Sigma, VT = np.linalg.svd(self.P)
            psi_k = phi @ U
            column_norms = np.linalg.norm(psi_k, axis=0)
            lamda_bar = np.sqrt(self.lamda) * np.eye(self.parameters,self.parameters)
            for i in range(len(column_norms)):
                if column_norms[i] <= self.e: 
                    lamda_bar[i, i] = 1 
            lamda_k = U@lamda_bar@U.T   
            pbar_k = np.linalg.inv(lamda_k)@self.P@np.linalg.inv(lamda_k)
            print('wS:',wS)
            print('phi:',phi)
            print('pbar_k:',pbar_k)
            print('prod:',np.linalg.inv(np.eye(phi.shape[0]) + wS*phi @ pbar_k @ phi.T))
            self.P = (pbar_k - pbar_k@(phi.T)@(np.linalg.inv(np.eye(phi.shape[0]) + wS*phi @ pbar_k @ phi.T))@phi@pbar_k)
            self.w = self.w + wS*self.P@(phi.T)@(y - phi@self.w) + self.P@((lamda_k@self.A@lamda_k - self.A)@(self.w_0 - self.w))
            self.A = lamda_k@self.A@lamda_k + phi.T@phi

        return
    
    def predict(self,X):
        #X = np.append(1,X)
        #print('X:',X)
        return self.w@X
    
    def Afine(self,m):
        L,C = m.shape
        col = np.ones(L).reshape(-1,1)
        m = np.hstack((col, m))
        return m