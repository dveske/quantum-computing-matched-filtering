# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:18:34 2023

@author: dvesk
"""


import numpy
import matplotlib.pyplot as plt
import numpy as np

div=20
a=numpy.genfromtxt('D:/GW190521/L-L1_GWOSC_4KHZ_R1-1242442952-32.txt/L-L1_GWOSC_4KHZ_R1-1242442952-32.txt')

bf=numpy.concatenate((numpy.fft.rfft(a)[:3276],numpy.zeros(len(numpy.fft.rfft(a))-3276)))#99.9755859375hz
b=numpy.fft.irfft(bf)
c=numpy.zeros(int(len(b)/div))
for i in range (0,int(len(b)/div)):
    c[i]=b[div*i]
psdt=plt.psd(c,Fs=4000/div,NFFT=2**9)
plt.xscale('log')
plt.close()
psd=numpy.interp(numpy.fft.rfftfreq(n=int(len(c)),d=div/4000),psdt[1],psdt[0])


wa=numpy.fft.irfft(numpy.fft.rfft(c)/psd)

wa=wa[int(15.15*len(wa)/32):int(15.6*len(wa)/32)]


wf=numpy.load('waveformlow.npy')#f_low=12Hz
wf1=wf[1].imag[:56]

wa=wa*10**-22.5
wf1=wf1*10**22.5


dd=numpy.zeros((int(numpy.ceil(len(wa)/3)),4))
datashift=numpy.zeros((int(numpy.ceil(len(wa)/3)),4))
data=numpy.zeros((int(numpy.ceil(len(wa)/3)),4))
for i in range (0,int(numpy.ceil(len(wa)/3))):
    if(len(wa[3*i:3*i+4])==4):
        dd[i]=wa[3*i:3*i+4]
    else:
        dd[i]=numpy.concatenate((wa[3*i:3*i+4],numpy.zeros(4-len(wa[3*i:3*i+4]))))
    datashift[i]=-min(dd[i])+1
    data[i]=(dd[i]+datashift[i])**0.5
    

signalshift=numpy.zeros(int(len(wf1)/2))
h_features1=numpy.zeros((int(len(wf1)/2),2))
h_features=numpy.zeros((int(len(wf1)/2),2))
for j in range (0,int(len(wf1)/2)):
    h_features1[j] = wf1[2*j:2*j+2]
    signalshift[j]=-min(h_features1[j])+1
    h_features[j] = (h_features1[j]+signalshift[j])**0.5
    h_features[j] = h_features[j] / np.linalg.norm(h_features[j])


countsa=numpy.load('countsa3.npy',allow_pickle=True)

look2=['00','10','10','01','01','11']

q=numpy.zeros((int(len(wf1)/2),int(numpy.ceil((len(wa)/3))),3))
cl=numpy.zeros((int(len(wf1)/2),int(numpy.ceil((len(wa)/3))),3))
qp=numpy.zeros((int(len(wf1)/2),int(numpy.ceil((len(wa))/3)),3))
clp=numpy.zeros((int(len(wf1)/2),int(numpy.ceil((len(wa))/3)),3))

corr=numpy.zeros((int(len(wf1)/2),int(numpy.ceil((len(wa))/3)),3))
for k in range (0,int(len(wf1)/2)):
    for j in range(0,int(numpy.ceil((len(wa)/3)))):
        if(j%3==0):
            for i in range (0,3):
                w_features = data[j][i:i+2]
                # normalize
                w_features = w_features / np.linalg.norm(w_features)
                w_features1 = data[j][i:i+2]**2-datashift[j][0]
                for kkk in countsa[int(numpy.ceil((len(wa)/9)))*k+int(j/3)].keys():
                    if((look2[2*i]==kkk[-2:] and kkk[-3]=='0') or (look2[2*i+1]==kkk[-2:] and kkk[-3]=='1')):
                        qp[k][j][i]+=countsa[int(numpy.ceil((len(wa)/9)))*k+int(j/3)][kkk]/(10000)
                q[k][j][i]=qp[k][j][i]*np.linalg.norm((h_features1[k]+signalshift[k])**0.5)**2*np.linalg.norm(data[j])**2-np.sum(datashift[j][0]*h_features1[k]+signalshift[k]*w_features1+datashift[j][0]*signalshift[k])
                clp[k][j][i]=(np.dot(dd[j][i:i+2],h_features1[k])+np.sum(datashift[j][0]*h_features1[k]+signalshift[k]*w_features1+datashift[j][0]*signalshift[k]))/(np.linalg.norm((h_features1[k]+signalshift[k])**0.5)**2*np.linalg.norm(data[j])**2)
                cl[k][j][i]=(np.dot(dd[j][i:i+2],h_features1[k]))
        elif (j%3==1):
            for i in range (0,3):
                w_features = data[j][i:i+2]
                # normalize
                w_features = w_features / np.linalg.norm(w_features)
                w_features1 = data[j][i:i+2]**2-datashift[j][0]
                for kkk in countsa[int(numpy.ceil((len(wa)/9)))*k+int(j/3)]:
                    if((look2[2*i]==kkk[-5:-3] and kkk[-3]=='0') or (look2[2*i+1]==kkk[-5:-3] and kkk[-3]=='1')):
                        qp[k][j][i]+=countsa[int(numpy.ceil((len(wa)/9)))*k+int(j/3)][kkk]/(10000)
                q[k][j][i]=qp[k][j][i]*np.linalg.norm((h_features1[k]+signalshift[k])**0.5)**2*np.linalg.norm(data[j])**2-np.sum(datashift[j][0]*h_features1[k]+signalshift[k]*w_features1+datashift[j][0]*signalshift[k])
                clp[k][j][i]=(np.dot(dd[j][i:i+2],h_features1[k])+np.sum(datashift[j][0]*h_features1[k]+signalshift[k]*w_features1+datashift[j][0]*signalshift[k]))/(np.linalg.norm((h_features1[k]+signalshift[k])**0.5)**2*np.linalg.norm(data[j])**2)
                cl[k][j][i]=(np.dot(dd[j][i:i+2],h_features1[k]))
        elif (j%3==2):
            for i in range (0,3):
                w_features = data[j][i:i+2]
                # normalize
                w_features = w_features / np.linalg.norm(w_features)
                w_features1 = data[j][i:i+2]**2-datashift[j][0]
                for kkk in countsa[int(numpy.ceil((len(wa)/9)))*k+int(j/3)]:
                    if((look2[2*i]==kkk[:2] and kkk[-3]=='0') or (look2[2*i+1]==kkk[:2] and kkk[-3]=='1')):
                        qp[k][j][i]+=countsa[int(numpy.ceil((len(wa)/9)))*k+int(j/3)][kkk]/(10000)
                q[k][j][i]=qp[k][j][i]*np.linalg.norm((h_features1[k]+signalshift[k])**0.5)**2*np.linalg.norm(data[j])**2-np.sum(datashift[j][0]*h_features1[k]+signalshift[k]*w_features1+datashift[j][0]*signalshift[k])
                clp[k][j][i]=(np.dot(dd[j][i:i+2],h_features1[k])+np.sum(datashift[j][0]*h_features1[k]+signalshift[k]*w_features1+datashift[j][0]*signalshift[k]))/(np.linalg.norm((h_features1[k]+signalshift[k])**0.5)**2*np.linalg.norm(data[j])**2)
                cl[k][j][i]=(np.dot(dd[j][i:i+2],h_features1[k]))
            
qrs=numpy.reshape(q,(int(len(wf1)/2),int(numpy.ceil((len(wa)/3)))*3))
crs=numpy.reshape(cl,(int(len(wf1)/2),int(numpy.ceil((len(wa)/3)))*3))
snr=numpy.zeros(len(wa)-len(wf1))
csnr=numpy.zeros(len(wa)-len(wf1))
for i in range (0,len(csnr)):
    snr[i]=numpy.sum([qrs[k][i+2*k] for k in range (0,int(len(wf1)/2))])
    csnr[i]=numpy.sum([crs[k][i+2*k] for k in range (0,int(len(wf1)/2))])
    


plt.rcParams.update({'font.size':20})
plt.rcParams.update({'lines.linewidth':2})
plt.plot(numpy.linspace(0,0.45,len(csnr)),csnr/max(abs(csnr)),marker='o',label='Classical',markerfacecolor='white',markersize=10)
plt.plot(numpy.linspace(0,0.45,len(csnr)),snr/max(abs(csnr)),marker='X',label='Quantum',markersize=8,linestyle='dashed')
plt.ylabel('Scaled SNR')
plt.xlabel('UTC time - 1242442967.15 [s]')
plt.grid()
plt.legend()
plt.tight_layout()