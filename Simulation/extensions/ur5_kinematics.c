#include <math.h>
#include <stdio.h>

#define PI M_PI
#define SIGN(x) ((x > 0) - (x < 0))
#define ZERO_THRESH 0.00000001

/* Copied from https://github.com/ros-industrial/universal_robot on 27 September 2019. */
extern "C" int InverseKinematicsUr5(const double* T, double* q_sols, double q6_des)
{
  // Constants for UR5
  const double d1 =  0.089159;
  const double a2 = -0.42500;
  const double a3 = -0.39225;
  const double d4 =  0.10915;
  const double d5 =  0.09465;
  const double d6 =  0.0823;
  
  // Initialization
  int num_sols = 0;
  double T02 = -*T; T++; double T00 =  *T; T++; double T01 =  *T; T++; double T03 = -*T; T++; 
  double T12 = -*T; T++; double T10 =  *T; T++; double T11 =  *T; T++; double T13 = -*T; T++; 
  double T22 =  *T; T++; double T20 = -*T; T++; double T21 = -*T; T++; double T23 =  *T;

  ////////////////////////////// shoulder rotate joint (q1) //////////////////////////////
  double q1[2];
  {
    double A = d6*T12 - T13;
    double B = d6*T02 - T03;
    double R = A*A + B*B;
    if(fabs(A) < ZERO_THRESH) {
      double div;
      if(fabs(fabs(d4) - fabs(B)) < ZERO_THRESH)
        div = -SIGN(d4)*SIGN(B);
      else
        div = -d4/B;
      double arcsin = asin(div);
      if(fabs(arcsin) < ZERO_THRESH)
        arcsin = 0.0;
      if(arcsin < 0.0)
        q1[0] = arcsin + 2.0*PI;
      else
        q1[0] = arcsin;
      q1[1] = PI - arcsin;
    }
    else if(fabs(B) < ZERO_THRESH) {
      double div;
      if(fabs(fabs(d4) - fabs(A)) < ZERO_THRESH)
        div = SIGN(d4)*SIGN(A);
      else
        div = d4/A;
      double arccos = acos(div);
      q1[0] = arccos;
      q1[1] = 2.0*PI - arccos;
    }
    else if(d4*d4 > R) {
      return num_sols;
    }
    else {
      double arccos = acos(d4 / sqrt(R)) ;
      double arctan = atan2(-B, A);
      double pos = arccos + arctan;
      double neg = -arccos + arctan;
      if(fabs(pos) < ZERO_THRESH)
        pos = 0.0;
      if(fabs(neg) < ZERO_THRESH)
        neg = 0.0;
      if(pos >= 0.0)
        q1[0] = pos;
      else
        q1[0] = 2.0*PI + pos;
      if(neg >= 0.0)
        q1[1] = neg; 
      else
        q1[1] = 2.0*PI + neg;
    }
  }
  ////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////// wrist 2 joint (q5) //////////////////////////////
  double q5[2][2];
  {
    for(int i=0;i<2;i++) {
      double numer = (T03*sin(q1[i]) - T13*cos(q1[i])-d4);
      double div;
      if(fabs(fabs(numer) - fabs(d6)) < ZERO_THRESH)
        div = SIGN(numer) * SIGN(d6);
      else
        div = numer / d6;
      double arccos = acos(div);
      q5[i][0] = arccos;
      q5[i][1] = 2.0*PI - arccos;
    }
  }
  ////////////////////////////////////////////////////////////////////////////////

  {
    for(int i=0;i<2;i++) {
      for(int j=0;j<2;j++) {
        double c1 = cos(q1[i]), s1 = sin(q1[i]);
        double c5 = cos(q5[i][j]), s5 = sin(q5[i][j]);
        double q6;
        ////////////////////////////// wrist 3 joint (q6) //////////////////////////////
        if(fabs(s5) < ZERO_THRESH)
          q6 = q6_des;
        else {
          q6 = atan2(SIGN(s5)*-(T01*s1 - T11*c1), 
                     SIGN(s5)*(T00*s1 - T10*c1));
          if(fabs(q6) < ZERO_THRESH)
            q6 = 0.0;
          if(q6 < 0.0)
            q6 += 2.0*PI;
        }
        ////////////////////////////////////////////////////////////////////////////////

        double q2[2], q3[2], q4[2];
        ///////////////////////////// RRR joints (q2,q3,q4) ////////////////////////////
        double c6 = cos(q6), s6 = sin(q6);
        double x04x = -s5*(T02*c1 + T12*s1) - c5*(s6*(T01*c1 + T11*s1) - c6*(T00*c1 + T10*s1));
        double x04y = c5*(T20*c6 - T21*s6) - T22*s5;
        double p13x = d5*(s6*(T00*c1 + T10*s1) + c6*(T01*c1 + T11*s1)) - d6*(T02*c1 + T12*s1) + 
                      T03*c1 + T13*s1;
        double p13y = T23 - d1 - d6*T22 + d5*(T21*c6 + T20*s6);

        double c3 = (p13x*p13x + p13y*p13y - a2*a2 - a3*a3) / (2.0*a2*a3);
        if(fabs(fabs(c3) - 1.0) < ZERO_THRESH)
          c3 = SIGN(c3);
        else if(fabs(c3) > 1.0) {
          // TODO NO SOLUTION
          continue;
        }
        double arccos = acos(c3);
        q3[0] = arccos;
        q3[1] = 2.0*PI - arccos;
        double denom = a2*a2 + a3*a3 + 2*a2*a3*c3;
        double s3 = sin(arccos);
        double A = (a2 + a3*c3), B = a3*s3;
        q2[0] = atan2((A*p13y - B*p13x) / denom, (A*p13x + B*p13y) / denom);
        q2[1] = atan2((A*p13y + B*p13x) / denom, (A*p13x - B*p13y) / denom);
        double c23_0 = cos(q2[0]+q3[0]);
        double s23_0 = sin(q2[0]+q3[0]);
        double c23_1 = cos(q2[1]+q3[1]);
        double s23_1 = sin(q2[1]+q3[1]);
        q4[0] = atan2(c23_0*x04y - s23_0*x04x, x04x*c23_0 + x04y*s23_0);
        q4[1] = atan2(c23_1*x04y - s23_1*x04x, x04x*c23_1 + x04y*s23_1);
        ////////////////////////////////////////////////////////////////////////////////
        for(int k=0;k<2;k++) {
          if(fabs(q2[k]) < ZERO_THRESH)
            q2[k] = 0.0;
          else if(q2[k] < 0.0) q2[k] += 2.0*PI;
          if(fabs(q4[k]) < ZERO_THRESH)
            q4[k] = 0.0;
          else if(q4[k] < 0.0) q4[k] += 2.0*PI;
          q_sols[num_sols*6+0] = q1[i];    q_sols[num_sols*6+1] = q2[k]; 
          q_sols[num_sols*6+2] = q3[k];    q_sols[num_sols*6+3] = q4[k]; 
          q_sols[num_sols*6+4] = q5[i][j]; q_sols[num_sols*6+5] = q6; 
          num_sols++;
        }

      }
    }
  }
  return num_sols;
}
