import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error

def cm(x):
    return x/2.54

def plot_RUL_CI(teda,startX=None,startY=None,endX=None,endY=None,
                anchor=None,w=6,h=4,out=None,name='Name',
                lw1=2,lw2=1.5,ftcks=7,flbl=8.5,fttl=8,flgnd=7,dotsize=2,
                loc=None,rect=[-0.06,-0.05,1.05,1],png=True,ncol=None):
    if ncol == None:
        ncol=1
        if teda.gCreated>4 : ncol=2
    
    if png:
        ext = '.png'
    else:
        ext = '.eps'

    if startX==None:  startX=teda.cycleP[int(np.where(teda.rulP==np.max(teda.rulP)-1)[0][0])]
    if endX==None:  endX=int(teda.eolX)+1
    if startY==None:  startY=0
    if endY==None:  endY=np.max(teda.rulP)-2.5
    activation = []

    for arr in teda.cloud_activation2:
        aux = np.array([None for j in range(teda.gCreated+1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T
    qtd = len(activation)-1

    names = [f'G{i+1}' for i in range(qtd)]
    xr,yr = [[] for i in range(qtd)],[[] for i in range(qtd)]
    
    if endX == 0:
        endX=x[-1]
    rulR = teda.rulR
    rulP = teda.rulP
    rulL = teda.rulL
    rulU = teda.rulU
    x = teda.cycleP
    for i in range(qtd):
        gran = activation[i+1]
        for l in range(len(gran)):
            if gran[l] ==i+1:
                yr[i].append(rulP[l])
                xr[i].append(x[l])
            
    plt.figure(figsize=(w, h))  # Define o tamanho do gráfico
    plt.plot(x, rulR, linestyle='-',linewidth=lw1, color='black', label="R-RUL")  # Plota os dados
    plt.plot(x, rulP, linestyle='-',linewidth=lw2, color='blue', label="P-RUL")
    plt.plot(x, rulU, linestyle='--',linewidth=lw1, color='blue',)
    plt.plot(x, rulL, linestyle='--',linewidth=lw1, color='blue',)
    plt.fill_between(x, rulL, rulU, color='skyblue', alpha=0.25, label="Uncertainty",)
    plt.fill_between(x, .8*rulR, 1.2*rulR, color='gray', alpha=0.15, label="T-20%",linewidth=lw1) #"IC:\u00B1 10%"
    plt.plot([teda.eolX,teda.eolX],[-1,np.max(teda.rulR)*1.2], color='black', linestyle=':', label='EoL')
    for i in range(len(xr)):
        #plt.plot(xr[i],yr[i],linestyle=' ', marker='o', markersize = dotsize,label = names[i],color = colors[i])
        plt.plot(xr[i],yr[i],linestyle=' ', marker='o', markersize = dotsize,label = names[i])
    plt.xlim(startX, endX+3)
    plt.ylim(startY, endY+1)
    plt.xticks( fontsize=ftcks, color='black')
    plt.yticks( fontsize=ftcks, color='black')
    plt.xlabel("Cycle",fontsize=flbl)  # Nome do eixo X
    plt.ylabel("RUL/Cycle",fontsize=flbl)  # Nome do eixo Y
    plt.title(f'{name} - Granular RUL Prediction',fontsize=fttl)  # Define o título do gráfico
    plt.grid(True,linewidth=0.5)  # Adiciona grade ao gráfico
    plt.legend(fontsize=flgnd,framealpha=0.85,loc = loc,bbox_to_anchor=anchor, ncol=ncol,columnspacing=0.7)
    plt.tight_layout(rect=rect) 
    if out != None and name != 'Name':
        plt.savefig(out+'RUL_'+name+ext, dpi=500,transparent=False)
    plt.show()  # Mostra o gráfico'

def plot_HI(teda,w=6,h=4,rect =[0,0,1,1],out=None,name=None,lnwdth=0.75,ftcks=7, flbl=8, fttl=8.5, 
            flgnd=7, anchor=None,png=True,  m1=1,m2=4,m3=1,ncol=None):
    if ncol==None:
        if teda.gCreated<5: ncol=1
        else: ncol=2
    if png:
        ext = '.png'
    else:
        ext = '.eps'
    x = teda.cycleP
    HI = teda.HI
    eolHI = teda.eol
    startX=teda.nI-2
    startY=0.15
    endX=teda.cycleP[-1]+2
    endY=np.max(teda.HI)*1.025
    activation = []
    for arr in teda.cloud_activation2:
        aux = np.array([None for j in range(teda.gCreated+1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T
    qtd = len(activation)-1
    names = [f'G{i+1}' for i in range(qtd)]
    xr,yr = [[] for i in range(qtd)],[[] for i in range(qtd)]

    for i in range(qtd):
        gran = activation[i+1]
        for l in range(len(gran)):
            if gran[l] ==i+1:
                yr[i].append(HI[l])
                xr[i].append(x[l])
 
    plt.figure(figsize=(w, h))
    for i in range(len(xr)):
        plt.plot(xr[i],yr[i],linestyle=' ',linewidth=lnwdth, marker='o', markersize = m1,label = names[i])
    plt.plot([0,endX],[eolHI,eolHI], color='black', linewidth=lnwdth, linestyle=':')
    plt.plot([teda.eolX,teda.eolX],[-1,1], color='black', linewidth=lnwdth, linestyle=':')
    plt.plot(teda.eolX, eolHI, marker='x', color='black', markersize=m2, linestyle='',markeredgewidth=m3, label='EOL')
    plt.xlim(startX, endX)
    plt.ylim(startY, endY)
    plt.xticks( fontsize=ftcks, color='black')
    plt.yticks( fontsize=ftcks, color='black')
    plt.xlabel("Cycle",fontsize=flbl)  # Nome do eixo X
    plt.ylabel("HI/Cycle",fontsize=flbl)  # Nome do eixo Y
    plt.title(f'{name} HI',fontsize=fttl)  # Define o título do gráfico
    plt.grid(False)  # Adiciona grade ao gráfico
    plt.legend(fontsize=flgnd,framealpha=0.85,loc = 'center left',bbox_to_anchor=anchor, ncol=ncol)
    plt.tight_layout(rect=rect) 
    if out != None and name != None:
        plt.savefig(out+'HI_'+name+ext, dpi=500,transparent=False)
    plt.show()  

def plot_DSI(teda,w=6,h=4,rect =[0,0,1,1],out=None,name=None,lnwdth=0.75,ftcks=7, flbl=8, fttl=8.5, 
            flgnd=7, anchor=None,png=True, m1=1,m2=4,m3=1,ncol=None):
    if ncol==None:
        if teda.gCreated<5: ncol=1
        else: ncol=2
    if png:
        ext = '.png'
    else:
        ext = '.eps'
    x = teda.cycleP
    DSI = teda.DSI
    eolDSI = teda.eolDSI
    startX=teda.nI-2
    startY=-np.min(teda.DSI)*5
    endX=teda.cycleP[-1]+2
    endY=np.max(teda.DSI)*1.025

    activation = []
    for arr in teda.cloud_activation2:
        aux = np.array([None for j in range(teda.gCreated+1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T
    qtd = len(activation)-1
    names = [f'G{i+1}' for i in range(qtd)]
    xr,yr = [[] for i in range(qtd)],[[] for i in range(qtd)]

    for i in range(qtd):
        gran = activation[i+1]
        for l in range(len(gran)):
            if gran[l] ==i+1:
                yr[i].append(DSI[l])
                xr[i].append(x[l])
 
    plt.figure(figsize=(w, h))
    for i in range(len(xr)):
        plt.plot(xr[i],yr[i],linestyle=' ',linewidth=lnwdth, marker='o', markersize = m1,label = names[i])
    plt.plot([0,endX],[eolDSI,eolDSI], color='black', linewidth=lnwdth, linestyle=':')
    plt.plot([teda.eolX,teda.eolX],[-1,1], color='black', linewidth=lnwdth, linestyle=':')
    plt.plot(teda.eolX, eolDSI, marker='x', color='black', markersize=m2, linestyle='',markeredgewidth=m3, label='EOL')
    plt.xlim(startX, endX)
    plt.ylim(startY, endY)
    plt.xticks( fontsize=ftcks, color='black')
    plt.yticks( fontsize=ftcks, color='black')
    plt.xlabel("Cycle",fontsize=flbl)  # Nome do eixo X
    plt.ylabel("DSI/Cycle",fontsize=flbl)  # Nome do eixo Y
    plt.title(f'{name} DSI',fontsize=fttl)  # Define o título do gráfico
    plt.grid(False)  # Adiciona grade ao gráfico
    plt.legend(fontsize=flgnd,framealpha=0.85,loc = 'upper right',bbox_to_anchor=anchor, ncol=ncol)
    plt.tight_layout(rect=rect) 
    if out != None and name != None:
        plt.savefig(out+'DSI_'+name+ext, dpi=500,transparent=False)
    plt.show()  

def plot_multiple_HI(teda_list, w=6, h=4, lnwdth=0.75, ftcks=7, flbl=8, fttl=8.5, 
            flgnd=7, m1=1, m2=4, m3=1, bearings=None, png=True, out=None, rect=[-0.025, 0.06, 1.02, 1.1]):
    n = len(teda_list)

    if png: ext = '.png'
    else: ext = '.eps'

    fig, axes = plt.subplots(1, n, figsize=(w, h), squeeze=False)

    # Identifica o teda com maior número de gCreated e seu índice
    max_idx = max(range(n), key=lambda i: teda_list[i].gCreated)
    teda_max = teda_list[max_idx]

    for idx, teda in enumerate(teda_list):
        ax = axes[0, idx]
        x = teda.cycleP
        DSI = teda.HI
        eolDSI = teda.eol
        startX = teda.nI-4
        startY = -0.05
        endX = teda.cycleP[-1] + 4
        endY = 1.03

        # Processa ativações
        activation = []
        for arr in teda.cloud_activation2:
            aux = np.array([None for _ in range(teda.gCreated + 1)])
            for k in range(len(arr)):
                aux[int(arr[k])] = int(arr[k])
            activation.append(aux)
        activation = np.array(activation).T
        qtd = len(activation) - 1
        names = [f'G{i + 1}' for i in range(qtd)]
        xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

        for i in range(qtd):
            gran = activation[i + 1]
            for l in range(len(gran)):
                if gran[l] == i + 1:
                    yr[i].append(DSI[l])
                    xr[i].append(x[l])

        # Plot individual
        for i in range(len(xr)):
            ax.plot(xr[i], yr[i], linestyle=' ', linewidth=lnwdth,
                    marker='o', markersize=m1, label=names[i])
        ax.plot([0, endX], [eolDSI, eolDSI], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot([teda.eolX, teda.eolX], [-1, 1], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot(teda.eolX, eolDSI, marker='x', color='black', markersize=m2, linestyle='',
                markeredgewidth=m3, label='EOL')

        ax.set_xlim(startX, endX)
        ax.set_ylim(startY, endY)
        ax.tick_params(axis='both', labelsize=ftcks)
        ax.set_xlabel("Cycle", fontsize=flbl)
        ax.set_ylabel("HI/Cycle", fontsize=flbl)
        if idx > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False,labelsize=5,axis='y',length=0.1,)
        ax.set_title(f'{bearings[idx]}', fontsize=fttl - 1)
        ax.grid(False)

    # Usar os handles reais da instância com maior gCreated
    ax_max = axes[0, max_idx]
    handles, labels = ax_max.get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=flgnd, framealpha=0.85,columnspacing=0.7)
    plt.suptitle('Granular HI', fontsize=fttl)
    plt.tight_layout(rect=rect)
    plt.subplots_adjust(wspace=0.05)

    if out is not None:
        plt.savefig(out + 'Bearings_HI' + ext, dpi=500, transparent=False)

    plt.show()

def plot_multiple_DSI(teda_list, w=6, h=4, lnwdth=0.75, ftcks=7, flbl=8, fttl=8.5, 
            flgnd=7, m1=1, m2=4, m3=1, bearings=None, png=True, out=None, rect=[-0.015, 0.06, 1.02, 1.1]):
    n = len(teda_list)

    if png: ext = '.png'
    else: ext = '.eps'

    fig, axes = plt.subplots(1, n, figsize=(w, h), squeeze=False)

    # Identifica o teda com maior número de gCreated e seu índice
    max_idx = max(range(n), key=lambda i: teda_list[i].gCreated)
    teda_max = teda_list[max_idx]

    for idx, teda in enumerate(teda_list):
        ax = axes[0, idx]
        x = teda.cycleP
        DSI = teda.DSI
        eolDSI = teda.eolDSI
        startX = teda.nI-3
        startY = -0.05
        endX = teda.cycleP[-1] + 4
        endY = 0.95

        # Processa ativações
        activation = []
        for arr in teda.cloud_activation2:
            aux = np.array([None for _ in range(teda.gCreated + 1)])
            for k in range(len(arr)):
                aux[int(arr[k])] = int(arr[k])
            activation.append(aux)
        activation = np.array(activation).T
        qtd = len(activation) - 1
        names = [f'G{i + 1}' for i in range(qtd)]
        xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

        for i in range(qtd):
            gran = activation[i + 1]
            for l in range(len(gran)):
                if gran[l] == i + 1:
                    yr[i].append(DSI[l])
                    xr[i].append(x[l])

        # Plot individual
        for i in range(len(xr)):
            ax.plot(xr[i], yr[i], linestyle=' ', linewidth=lnwdth,
                    marker='o', markersize=m1, label=names[i])
        ax.plot([0, endX], [eolDSI, eolDSI], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot([teda.eolX, teda.eolX], [-1, 1], color='black', linewidth=lnwdth, linestyle=':')
        ax.plot(teda.eolX, eolDSI, marker='x', color='black', markersize=m2, linestyle='',
                markeredgewidth=m3, label='EOL')

        ax.set_xlim(startX, endX)
        ax.set_ylim(startY, endY)
        ax.tick_params(axis='both', labelsize=ftcks)
        ax.set_xlabel("Cycle", fontsize=flbl)
        ax.set_ylabel("DSI/Cycle", fontsize=flbl)
        if idx > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False,labelsize=5,axis='y',length=0.1,)
        ax.set_title(f'{bearings[idx]}', fontsize=fttl - 1)
        ax.grid(False)

    # Usar os handles reais da instância com maior gCreated
    ax_max = axes[0, max_idx]
    handles, labels = ax_max.get_legend_handles_labels()

    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=flgnd, framealpha=0.85,columnspacing=0.7)
    plt.suptitle('Granular DSI', fontsize=fttl)
    plt.tight_layout(rect=rect)
    plt.subplots_adjust(wspace=0.05)

    if out is not None:
        plt.savefig(out + 'Bearings_DSI' + ext, dpi=500, transparent=False)

    plt.show()


def metrics(teda, w=6, h=4, out=None, name=None, lnwdth=2, ftcks=14, flbl=16, fttl=18, flgnd=14):
    rulR = np.array(teda.rulR)
    rulP = np.array(teda.rulP)

    rmse = []
    mae = []
    for i in range(1, len(rulR)):  # Começa de 1 para evitar vetores vazios
        rmse.append(np.sqrt(mean_squared_error(rulR[:i], rulP[:i])))
    for i in range(1,len(rulR)+1):
      mae.append((mean_absolute_error(rulR[:i], rulP[:i])))
    
    # Cria uma figura com 2 gráficos lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(w, h))

    # Primeiro gráfico: RMSE
    axes[0].plot(rmse, linewidth=lnwdth)
    axes[0].set_title(f'{name[:-4]}: \n RUL Prediction RMSE',fontsize=fttl)
    axes[0].set_xlabel('Cycle', fontsize=flbl)
    axes[0].set_ylabel('RMSE', fontsize=flbl)
    axes[0].grid(True, linestyle='--', linewidth=0.5)
    axes[0].tick_params(labelsize=ftcks)

    # Segundo gráfico: rulR vs rulP
    axes[1].plot(mae, linewidth=lnwdth)
    axes[1].set_title(f'{name[:-4]}: \n RUL Prediction MAE',fontsize=fttl)
    axes[1].set_xlabel('Cycle', fontsize=flbl)
    axes[1].set_ylabel('MAE', fontsize=flbl)
    axes[1].grid(True, linestyle='--', linewidth=0.5)
    axes[1].tick_params(labelsize=ftcks)

    plt.tight_layout()

    # Salva se desejar
    if out is not None and name is not None:
        plt.savefig(out + 'Performance_'+name, dpi=500)
    
    plt.show()

def metric2(teda, w=6, h=4, out=None, name=None, lnwdth=2, ftcks=14, flbl=16, fttl=18, flgnd=14):
    rulR = np.array(teda.rulR)
    rulP = np.array(teda.rulP)

    rmse = []
    mae = []
    for i in range(1, len(rulR)):  # Começa de 1 para evitar vetores vazios
        rmse.append(np.sqrt(mean_squared_error(rulR[:i], rulP[:i])))
    for i in range(1,len(rulR)+1):
      mae.append((mean_absolute_error(rulR[:i], rulP[:i])))
    
    # Cria uma figura com 2 gráficos lado a lado
    plt.figure(figsize=(w, h))

    # Primeiro gráfico: RMSE
    plt.plot(rmse,color='blue',linewidth=lnwdth,label='RMSE')
    plt.plot(mae,color='red',linewidth=lnwdth,label='MAE')

    plt.xticks( fontsize=ftcks, color='black')
    plt.yticks( fontsize=ftcks, color='black')
    plt.xlabel("Cycle",fontsize=flbl)  # Nome do eixo X
    plt.ylabel("Metric Value",fontsize=flbl)  # Nome do eixo Y

    plt.title(f'{name[:-4]}:\nPerformance',fontsize=fttl)  # Define o título do gráfico
    plt.grid(True)  # Adiciona grade ao gráfico
    plt.legend(fontsize=flgnd,framealpha=1,loc='upper right')
    plt.tight_layout()

    # Salva se desejar
    if out is not None and name is not None:
        plt.savefig(out + 'Performance_'+name, dpi=500)
    
    plt.show()

def _plot_RUL_CI_on_ax(ax, teda,startX=None, name='Name', lw1=1, lw2=0.75, ftcks=7, flbl=8, fttl=8, flgnd=7, m1=1, m2=5, m3=1.5):
    activation = []
    for arr in teda.cloud_activation2:
        aux = np.array([None for _ in range(teda.gCreated+1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T
    qtd = len(activation) - 1
    names = [f'G{i+1}' for i in range(qtd)]
    xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

    eolRUL = teda.rulP[int(teda.eolX - teda.nI)]
    rulR, rulP, rulL, rulU = teda.rulR, teda.rulP, teda.rulL, teda.rulU
    x = teda.cycleP

    for i in range(qtd):
        gran = activation[i+1]
        for l in range(len(gran)):
            if gran[l] == i+1:
                yr[i].append(rulP[l])
                xr[i].append(x[l])

    if startX==None:
        startX = teda.cycleP[int(np.where(teda.rulP == np.max(teda.rulP) - 1)[0][0])]+1
    endX = int(teda.eolX)
    startY = 0
    endY = np.max(teda.rulP) - 2.5

    ax.plot(x, rulR, '-', linewidth=lw1, color='black', label="Real RUL")
    ax.plot(x, rulP, '-', linewidth=lw2, color='blue', label="Predicted RUL")
    ax.plot(x, rulU, '--', linewidth=lw1, color='blue')
    ax.plot(x, rulL, '--', linewidth=lw1, color='blue')
    ax.fill_between(x, 0.8 * rulR, 1.2 * rulR, color='skyblue', alpha=0.15, label="Tolerance 20%")
    ax.fill_between(x, rulL, rulU, color='lightgray', alpha=0.25, label="Uncertainty")

    for i in range(len(xr)):
        ax.plot(xr[i], yr[i], linestyle=' ', marker='o', markersize=m1*2, label=names[i])

    ax.plot(teda.eolX, eolRUL, marker='x', color='black', markersize=m2, linestyle='', markeredgewidth=m3, label='EOL')
    ax.plot([teda.eolX, teda.eolX], [-200, +200], color='black', linewidth=lw2, linestyle=':')
    ax.plot([-200, +200], [eolRUL, eolRUL], color='black', linewidth=lw2, linestyle=':')
    ax.set_xlim(startX, endX + 1)
    ax.set_ylim(startY, endY + 1)
    ax.tick_params(axis='both', labelsize=ftcks, colors='black')
    ax.set_xlabel("Cycle", fontsize=flbl)
    ax.set_ylabel("RUL/Cycle", fontsize=flbl)
    ax.set_title(f'{name}', fontsize=fttl)
    ax.grid(True, linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def _plot_HI_on_ax(ax, teda, name=None, lnwdth=0.75, ftcks=7, flbl=8, fttl=8, flgnd=7, m1=1, m2=5, m3=1.5):
    x = teda.cycleP
    HI = teda.HI
    eolHI = teda.eol
    startX = teda.nI - 2
    startY = 0.15
    endX = x[-1] + 2
    endY = np.max(HI) * 1.025

    activation = []
    for arr in teda.cloud_activation2:
        aux = np.array([None for _ in range(teda.gCreated + 1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T

    qtd = len(activation) - 1
    names = [f'G{i+1}' for i in range(qtd)]
    xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

    for i in range(qtd):
        gran = activation[i+1]
        for l in range(len(gran)):
            if gran[l] == i+1:
                yr[i].append(HI[l])
                xr[i].append(x[l])

    for i in range(len(xr)):
        ax.plot(xr[i], yr[i], linestyle=' ', linewidth=lnwdth, marker='o', markersize=m1, label=names[i])

    ax.plot([0, endX], [eolHI, eolHI], color='black', linewidth=lnwdth, linestyle=':')
    ax.plot([teda.eolX, teda.eolX], [-1, 1], color='black', linewidth=lnwdth, linestyle=':')
    ax.plot(teda.eolX, eolHI, marker='x', color='black', markersize=m2, linestyle='', markeredgewidth=m3, label='EOL')

    ax.set_xlim(startX, endX)
    ax.set_ylim(startY, endY)
    ax.tick_params(axis='both', labelsize=ftcks, colors='black')
    ax.set_xlabel("Cycle", fontsize=flbl)
    ax.set_ylabel("HI/Cycle", fontsize=flbl)
    ax.set_title(f'{name}', fontsize=fttl)
    ax.grid(False)


def _plot_DSI_on_ax(ax, teda, name=None, lnwdth=0.75, ftcks=7, flbl=8, fttl=8, flgnd=7, m1=1, m2=5, m3=1.5):
    x = teda.cycleP
    DSI = teda.DSI
    eolDSI = teda.eolDSI
    startX = teda.nI - 2
    startY = np.min(DSI) -0.025
    endX = x[-1] + 2
    endY = np.max(DSI) * 1.025

    activation = []
    for arr in teda.cloud_activation2:
        aux = np.array([None for _ in range(teda.gCreated + 1)])
        for k in range(len(arr)):
            aux[int(arr[k])] = int(arr[k])
        activation.append(aux)
    activation = np.array(activation).T

    qtd = len(activation) - 1
    names = [f'G{i+1}' for i in range(qtd)]
    xr, yr = [[] for _ in range(qtd)], [[] for _ in range(qtd)]

    for i in range(qtd):
        gran = activation[i+1]
        for l in range(len(gran)):
            if gran[l] == i+1:
                yr[i].append(DSI[l])
                xr[i].append(x[l])

    for i in range(len(xr)):
        ax.plot(xr[i], yr[i], linestyle=' ', linewidth=lnwdth, marker='o', markersize=m1, label=names[i])

    ax.plot([0, endX], [eolDSI, eolDSI], color='black', linewidth=lnwdth, linestyle=':')
    ax.plot([teda.eolX, teda.eolX], [-1, 1], color='black', linewidth=lnwdth, linestyle=':')
    ax.plot(teda.eolX, eolDSI, marker='x', color='black', markersize=m2, linestyle='', markeredgewidth=m3, label='EOL')

    ax.set_xlim(startX, endX)
    ax.set_ylim(startY, endY)
    ax.tick_params(axis='both', labelsize=ftcks, colors='black')
    ax.set_xlabel("Cycle", fontsize=flbl)
    ax.set_ylabel("DSI/Cycle", fontsize=flbl)
    ax.set_title(f'{name}', fontsize=fttl)
    ax.grid(False)


def plot_RUL_HI_DSI_side_by_side(teda,startX=None, name="Name", w=cm(14), h=cm(6), out=None, png=True, rect=[-0.025, 0.09, 1.025, 1.08]):
    fig, axes = plt.subplots(1, 3, figsize=(w, h), gridspec_kw={'width_ratios': [1.5, 1.5, 3]})

    # Plot subplots e captura da legenda apenas do primeiro
    _plot_DSI_on_ax(axes[0], teda, name='a) Granular DSI')
    _plot_HI_on_ax(axes[1], teda, name='b) Granular HI')
    handles, labels = _plot_RUL_CI_on_ax(axes[2], teda, startX=startX,name='c) Granular RUL prediction')

    plt.suptitle(name, fontsize=10)

    # Legenda horizontal centralizada abaixo da figura
    if len(labels)< 8: ncol=len(labels)
    else: ncol=7
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01),
               ncol=ncol, fontsize=7, framealpha=0.85,
               columnspacing=0.5, handletextpad=0.5)

    plt.tight_layout(rect=rect)
    plt.subplots_adjust(wspace=0.35)
    if out is not None:
        ext = '.png' if png else '.eps'
        plt.savefig(out + name + ext, dpi=500, transparent=True)

    plt.show()

def plot_TS(teda,startX=None,startY=None,endX=None,endY=None,
                anchor=None,w=cm(10),h=cm(5),out=None,name='Name',
                lw1=1.25,lw2=0.5,lw3=0.75,ftcks=7,flbl=8,fttl=8,flgnd=7,dotsize=2,
                loc='lower center',rect=[-0.03,-0.06,1.02,1.06],
                png=False,ncol=None,plt_L=True,plt_U=True,plt_P=True,plt_R=True):
    
    if startX==None:  startX=0
    if endX==None:  endX=len(teda.HIp)
    if startY==None:  startY=np.min(teda.HIpL)*0.95
    if endY==None:  endY=np.max(teda.HIpU)*1.05

    plt.figure(figsize=(w, h)) 
    if plt_R: plt.plot(teda.cycleP, teda.HI, linestyle='-',linewidth=lw1, color='Black', label="Real")
    if plt_P: plt.plot(teda.cycleP, teda.HIp, linestyle='--',linewidth=lw2, color='red', label="Predicted")
    if plt_L: plt.plot(teda.cycleP, teda.HIpL, linestyle='-',linewidth=lw3, color='green', label="Lower bound")
    if plt_U: plt.plot(teda.cycleP, teda.HIpU, linestyle='-',linewidth=lw3, color='blue', label="Upper bound")
    plt.xlim(startX, endX)
    plt.ylim(startY, endY)
    plt.xticks( fontsize=ftcks, color='black')
    plt.yticks( fontsize=ftcks, color='black')
    plt.xlabel("Cycle",fontsize=flbl)  # Nome do eixo X
    plt.ylabel("RUL/Cycle",fontsize=flbl)  # Nome do eixo Y
    plt.title(f'{name} - Time series prediction',fontsize=fttl)  # Define o título do gráfico
    plt.grid(True,linewidth=0.5)  # Adiciona grade ao gráfico
    plt.legend(fontsize=flgnd,framealpha=0.85,loc = loc,bbox_to_anchor=anchor, ncol=2,columnspacing=0.5)
    plt.tight_layout(rect=rect) 
    if out is not None:
        ext = '.png' if png else '.eps'
        plt.savefig(out + name + ext, dpi=500, transparent=True)
    plt.show()

