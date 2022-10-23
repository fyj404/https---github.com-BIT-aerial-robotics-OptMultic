#include <ceres/ceres.h>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include<yaml-cpp/yaml.h>
#include<Eigen/Dense>
#include<Eigen/Core>
#include<bits/stdc++.h>
#include "matplotlibcpp.h"
using namespace std;
namespace plt=matplotlibcpp;

const int SIZE_POSE=7;
class SE3PoseLocalParameterization : public ceres::LocalParameterization {
public:
    SE3PoseLocalParameterization() {}
    virtual ~SE3PoseLocalParameterization() {}

    /**
     * @brief: GOBAL：SE3， LOCAL ：se3 
     * @param x : SE3 四元数 + XYZ 
     * @param delta se3 
     * @param x_plus_delta  SE3 
     */    
    virtual bool Plus(const double *x,  const double *delta, double *x_plus_delta) const 
    {
        Eigen::Map<const Eigen::Vector3d> _p(x+4);
        Eigen::Map<const Eigen::Quaterniond> _q(x);   // 传入序列 (x, y, z, w)构造eigen 四元数   注意eigen 内部存储四元数的顺序是  (x, y, z, w) 
        Eigen::Quaterniond delta_q;  
        Eigen::Vector3d delta_t;  
        Eigen::Map<const Eigen::Vector3d> delta_se3_omega(delta);    // se3中旋转部分
        Eigen::Map<const Eigen::Vector3d> delta_se3_upsilon(delta+3);    // se3中t    

        Eigen::Vector3d unit_delta_se3_omega = delta_se3_omega.normalized(); 
        double theta = delta_se3_omega.norm();                            

        if(theta<1e-10)
        {
            delta_q = Eigen::Quaterniond(1, delta_se3_omega.x() / 2, 
                                            delta_se3_omega.y() / 2, 
                                            delta_se3_omega.z() / 2);
        }
        else
        {
            double sin_half_theta = sin(0.5*theta);
            delta_q = Eigen::Quaterniond(cos(0.5*theta), unit_delta_se3_omega.x()*sin_half_theta, 
                                          unit_delta_se3_omega.y()*sin_half_theta, 
                                          unit_delta_se3_omega.z()*sin_half_theta);
        }

        Eigen::Matrix3d J;

        if (theta<1e-10)
        {
            J = delta_q.matrix();   // 当theta很小时， 近似为罗德里格斯公式 
        }
        else
        {
            double c = sin(theta) / theta;
            J = Eigen::Matrix3d::Identity()*c + (1 - c)*unit_delta_se3_omega*unit_delta_se3_omega.transpose() + 
                    (1 - cos(theta))*Utility::getSkewMatrix(unit_delta_se3_omega) / theta ; 
        }

        delta_t = J*delta_se3_upsilon;  
        Eigen::Map<Eigen::Vector3d> p(x_plus_delta+4);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta);
        q = (delta_q*_q).normalized();
        p = delta_q * _p + delta_t; 
        return true;
    }

    /
    virtual bool ComputeJacobian(const double *x, double *jacobian) const 
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
        (j.topRows(6)).setIdentity();
        (j.bottomRows(1)).setZero();

        return true;
    }

    virtual int GlobalSize() const { return 7;};
    virtual int LocalSize() const { return 6; };
};
YAML::Node config;
int node_number=0;
vector<Eigen::Matrix4d>Rt_Cam(10,Eigen::MatrixXd::Zero(4,4));
string uavName[10];
vector<vector<double>>dataCsv[10];
void show(vector<vector<double > > &strArray,int idx)
{
    int n=strArray.size();
    vector<double>x(n);
    vector<double>y(n);
    for(int i=0;i<n;i++){
        x.at(i)=i;
        y.at(i)=strArray[i][idx];
    }
    //plt::plot(x,y);
    //plt::show();
}

int lower_find(vector<vector<double > > &strArray,double time)
{
    int L=0,R=strArray.size()-1,ans=strArray.size()-1;
    while(L<=R){
        int mid=(L+R)>>1;
        if(strArray[mid][0]>=time){
            R=mid-1;
            ans=mid;
        }
        else{
            L=mid;
        }
    }return ans;
}

double Lagrange(vector<vector<double > > &strArray,int L,int R,int idx,double inputX)
{
    double res=0;
    double MultiX[200];
    double t1=0,t2=0;
    int i,j;
    for(i=L;i<=R;i++){
        t1=1,t2=1;
        for(j=L;j<=R;j++){
            if(i==j)continue;
            t1=t1*(inputX-strArray[j][0]);
            t2=t2*(strArray[i][0]-strArray[j][0]);
        }
        MultiX[i-L]=t1/t2;
    }
    for(i=L;i<=R;i++){
        res+=strArray[i][idx]*MultiX[i-L];
    }
    return res;
}
void slerp(vector<double>p,vector<double>q,double res[4],double t)
{
    double cosA=p[4]*q[4]+p[5]*q[5]+p[6]*q[6]+p[7]*q[7];
    if(cosA<-0.0){
        q[4]*=-1;
        q[5]*=-1;
        q[6]*=-1;
        q[7]*=-1;
        cosA*=-1;
    }
    double k0,k1;
    if(cosA>0.999995){
        k0=1.0-t;
        k1=t;
    }
    else{
        double sinA=sqrt(1-cosA*cosA);
        double A=atan2(sinA,cosA);
        k0=sin((1.0-t)*A)/sinA;
        k1=sin(t*A)/sinA;
    }
    res[0]=p[4]*k0+q[4]*k1;
    res[1]=p[5]*k0+q[5]*k1;
    res[2]=p[6]*k0+q[6]*k2;
    res[3]=p[7]*k0+q[7]*k3;

    double len=sqrt(res[0]*res[0]+res[1]*res[1]+res[2]*res[2]+res[3]*res[3]);
    res[0]/=len;
    res[1]/=len;
    res[2]/=len;
    res[3]/=len;
}
void read_csv(string fileName,vector<vector<double > > &strArray)
{
    ifstream inFile(fileName);
    string lineStr;
    cout<<fileName<<endl;
    while(getline(inFile,lineStr))
    {
        stringstream ss(lineStr);
        string str;
        vector<double>lineArray;
        while(getline(ss,str,',')){
            lineArray.push_back(atof(str.c_str()));
        }
        if(lineArray[0]>100000000000000){
            lineArray[0]/=1e9;
        }
        strArray.push_back(lineArray);
    }
    cout<<strArray.size()<<endl;
}

void sol_data_to_NED(vector<vector<double > > &strArray,int idx)
{

    int n=strArray.size();
    int m=strArray[0].size();
    Eigen::Matrix4d rt=Rt_Cam[idx];
    for(int i=0;i<n;i++){

        Eigen::Vector4d pos;
        pos(0)=strArray[i][1];
        pos(1)=strArray[i][2];
        pos(2)=strArray[i][3];
        pos(3)=1;
        Eigen::Vector4d newPos=rt*pos;
        strArray[i][1]=newPos(0);
        strArray[i][2]=newPos(1);
        strArray[i][3]=newPos(2);
        if(i<10)
        cout<<strArray[i][1]<<" "<<strArray[i][2]<<"  "<<strArray[i][3]<<endl;
    }
    for(int i=1;i<=3;i++)show(strArray,i);
}
void sol_data_to_TIME(vector<vector<double > > &strArray)
{

    vector<vector<double > > v(strArray);

    strArray.clear();
    strArray.resize(0);
    int N=dataCsv[0].size();
    for(int i=0;i<N;i++){

        double time=dataCsv[0][i][0];
        int low_idx=lower_find(v,time);
        int mn=max(0,low_idx-5);
        int mx=min(v.size()-1,low_idx+5);
        vector<double> tmp(11,0);
        tmp[0]=time;
        for(int j=1;j<=3;j++)
        {
            double val=Lagrange(v,mn,mx,j,time);
            tmp[j]=val;
        }
        mn=max(0,low_idx-1);
        mx=min(v.size()-1,low_idx+1);
        double t=(time-v[mn])/(v[mx]-v[mn]);
        double tmpQuad[4]={0,0,0,0};
        slerp(v[mn],v[mx],tmpQuad,t);
        tmp[4]=tmpQuad[0];
        tmp[5]=tmpQuad[1];
        tmp[6]=tmpQuad[2];
        tmp[7]=tmpQuad[3];
        strArray.push_back(tmp);
    }
}
void apply_SMCKF()
{

    double para_Pose[20][20];
    for(int i=0;i<dataCsv[0].size();i++){
        ceres::Problem problem;
        ceres::LossFunction *loss_function;                           // 损失核函数 
        //loss_function = new ceres::HuberLoss(1.0);
        loss_function = new ceres::CauchyLoss(1.0);                   // 柯西核函数   
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        for(int j=1;j<=node_number;j++)
            for(int k=0;k<=6;k++)para_Pose[j][k]=dataCsv[j][i][k+1]; //warning 顺序
            problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
    }
}
void eraseBeginPoint()
{
    double minT=dataCsv[1][0][0];
    for(int i=2;i<=node_number;i++)minT=min(minT,dataCsv[i][0][0]);
    while(dataCsv[0].size()>1&&dataCsv[0][0][0]<minT)dataCsv[0].erase(dataCsv[0].begin());
}
int main(int argc, char const* argv[])
{
    std::string yamlPath;
    if(argc>1){
        yamlPath="../config/"+std::string(argv[1]);
       
    }
    else{
        yamlPath=std::string("../config/lab1.yaml");
    }
    std::cout<<"yamlPath:   "+yamlPath<<std::endl;
    
    config=YAML::LoadFile(yamlPath);
    node_number=config["uav_number"].as<int>();
    string lab_number=config["lab_number"].as<string>();
    std::cout<<node_number<<endl;


    string gt_path="../data/"+config["gt"]["name"].as<string>()+"/lab"+lab_number+"/gt.csv";;
    read_csv(uav_data_path_name,dataCsv[0]);


    for(int i=1;i<=node_number;i++){
        std::vector<double> ext=config["uav_"+std::to_string(i)]["body_T_cam"]["data"].as<std::vector<double>>();
        for(int j=0;j<4;j++){
            for(int k=0;k<4;k++){
                Rt_Cam[i](j,k)=ext[j*4+k];
            }
        }
        uavName[i]=config["uav_"+std::to_string(i)]["name"].as<string>();
        std::string uav_data_path_name="../data/"+uavName[i]+"/lab"+lab_number+"/vio.csv";
        read_csv(uav_data_path_name,dataCsv[i]);
    }
    
    eraseBeginPoint();
    for(int i=1;i<=node_number;i++){
        sol_data_to_NED(dataCsv[i],i);
        sol_data_to_TIME(dataCsv[i]);
    }


    apply_SMCKF();
    return 0;
}

