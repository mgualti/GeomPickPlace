/* Furthest point sampling GPU implementation
 * Original author: Haoqiang Fan
 * Modified by Charles R. Qi
 * All Rights Reserved. 2017.
 */

# include<iostream>
# include<algorithm>

using namespace std;

void cumsumKernel(int b, int n, const float* inp, float* out)
{
  const int BlockSize=2048;
  const int paddingLevel=5;
  float buffer4[BlockSize*4];
  float buffer[BlockSize+(BlockSize>>paddingLevel)];

  for (int i=0; i<b; i++)
  {
    float runningsum=0,runningsum2=0;
    for (int j=0;j<n;j+=BlockSize*4)
    {
      int n24_i=min(n-j,BlockSize*4);
      int n24=(n24_i+3)&~3;
      int n2=n24>>2;
      for (int k=0*4;k<n24_i;k+=4){
        if (k+3<n24_i){
          float v1=inp[i*n+j+k];
          float v2=inp[i*n+j+k+1];
          v2+=v1;
          float v3=inp[i*n+j+k+2];
          float v4=inp[i*n+j+k+3];
          v4+=v3;
          v3+=v2;
          v4+=v2;
          buffer4[k]=v1;
          buffer4[k+1]=v2;
          buffer4[k+2]=v3;
          buffer4[k+3]=v4;
          buffer[(k>>2)+(k>>(2+paddingLevel))]=v4;
        }else{
          float v=0;
          for (int k2=k;k2<n24_i;k2++){
            v+=inp[i*n+j+k2];
            buffer4[k2]=v;
          }
          for (int k2=n24_i;k2<n24;k2++){
            buffer4[k2]=v;
          }
          buffer[(k>>2)+(k>>(2+paddingLevel))]=v;
        }
      }
      int u=0;
      for (;(2<<u)<=n2;u++){
        for (int k=0;k<int(n2>>(u+1));k++){
          int i1=(((k<<1)+2)<<u)-1;
          int i2=(((k<<1)+1)<<u)-1;
          i1+=i1>>paddingLevel;
          i2+=i2>>paddingLevel;
          buffer[i1]+=buffer[i2];
        }
      }
      u--;
      for (;u>=0;u--){
        for (int k=0;k<int((n2-(1<<u))>>(u+1));k++){
          int i1=(((k<<1)+3)<<u)-1;
          int i2=(((k<<1)+2)<<u)-1;
          i1+=i1>>paddingLevel;
          i2+=i2>>paddingLevel;
          buffer[i1]+=buffer[i2];
        }
      }
      for (int k=0*4;k<n24;k+=4){
        if (k!=0){
          int k2=((k>>2)-1)+(((k>>2)-1)>>paddingLevel);
          buffer4[k]+=buffer[k2];
          buffer4[k+1]+=buffer[k2];
          buffer4[k+2]+=buffer[k2];
          buffer4[k+3]+=buffer[k2];
        }
      }
      for (int k=0;k<n24_i;k++){
        out[i*n+j+k]=buffer4[k]+runningsum;
      }
      float t=buffer[(n2-1)+((n2-1)>>paddingLevel)]+runningsum2;
      float r2=runningsum+t;
      runningsum2=t-(r2-runningsum);
      runningsum=r2;
    }
  }
}

void binarysearchKernel(int b, int n, int m, const float* dataset, const float* query, int* result)
{
  int base=1;
  while (base<n)
    base<<=1;
  for (int i=0;i<b;i++)
  {
    for (int j=0;j<m;j++)
    {
      float q=query[i*m+j]*dataset[i*n+n-1];
      int r=n-1;
      for (int k=base;k>=1;k>>=1)
        if (r>=k && dataset[i*n+r-k]>=q)
          r-=k;
      result[i*m+j]=r;
    }
  }
}
void farthestpointsamplingKernel(int b, int n, int m, const float* dataset, float* temp, int* idxs)
{
  if (m<=0) return;
  const int BlockSize=512;
  float dists[BlockSize];
  int dists_i[BlockSize];
  const int BufferSize=3072;
  float buf[BufferSize*3];
  for (int i=0;i<b;i++){
    int old=0;
    idxs[i*m+0]=old;
    for (int j=0;j<n;j++){
      temp[0*n+j]=1e38;
    }
    for (int j=0;j<min(BufferSize,n)*3;j++){
      buf[j]=dataset[i*n*3+j];
    }
    for (int j=1;j<m;j++){
      int besti=0;
      float best=-1;
      float x1=dataset[i*n*3+old*3+0];
      float y1=dataset[i*n*3+old*3+1];
      float z1=dataset[i*n*3+old*3+2];
      for (int k=0;k<n;k++){
        float td=temp[0*n+k];
        float x2,y2,z2;
        if (k<BufferSize){
          x2=buf[k*3+0];
          y2=buf[k*3+1];
          z2=buf[k*3+2];
        }else{
          x2=dataset[i*n*3+k*3+0];
          y2=dataset[i*n*3+k*3+1];
          z2=dataset[i*n*3+k*3+2];
        }
        float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
        float d2=min(d,td);
        if (d2!=td)
          temp[0*n+k]=d2;
        if (d2>best){
          best=d2;
          besti=k;
        }
      }
      dists[0]=best;
      dists_i[0]=besti;
      for (int u=0;(1<<u)<1;u++){
        if (0<(1>>(u+1))){
          int i1=(0*2)<<u;
          int i2=(0*2+1)<<u;
          if (dists[i1]<dists[i2]){
            dists[i1]=dists[i2];
            dists_i[i1]=dists_i[i2];
          }
        }
      }
      old=dists_i[0];
      idxs[i*m+j]=old;
    }
  }
}

void gatherpointKernel(int b,int n,int m,const float* inp, const int* idx, float* out)
{
  for (int i = 0; i < b; i++)
  {
    for (int j = 0; j < m; j++)
    {
      int a=idx[i*m+j];
      out[(i*m+j)*3+0]=inp[(i*n+a)*3+0];
      out[(i*m+j)*3+1]=inp[(i*n+a)*3+1];
      out[(i*m+j)*3+2]=inp[(i*n+a)*3+2];
    }
  }
}

void scatteraddpointKernel(int b, int n, int m, const float* out_g, const int* idx, float* inp_g)
{
  for (int i = 0; i < b; i++)
  {
    for (int j = 0; j < m; j++)
    {
      int a = idx[i * m+j];
      inp_g[(i*n+a)*3+0] += out_g[(i*m+j)*3+0];
      inp_g[(i*n+a)*3+1] += out_g[(i*m+j)*3+1];
      inp_g[(i*n+a)*3+2] += out_g[(i*m+j)*3+2];
    }
  }
}

//require b*n working space
void probsampleLauncher(int b,int n,int m,const float * inp_p,const float * inp_r,float * temp,int * out){
  cumsumKernel(b,n,inp_p,temp);
  binarysearchKernel(b,n,m,temp,inp_r,out);
}
//require 32*n working space
void farthestpointsamplingLauncher(int b,int n,int m,const float * inp,float * temp,int * out){
  farthestpointsamplingKernel(b,n,m,inp,temp,out);
}
void gatherpointLauncher(int b,int n,int m,const float * inp,const int * idx,float * out){
  gatherpointKernel(b,n,m,inp,idx,out);
}
void scatteraddpointLauncher(int b,int n,int m,const float * out_g,const int * idx,float * inp_g){
  scatteraddpointKernel(b,n,m,out_g,idx,inp_g);
}

