#ifndef ANN_H
#define ANN_H


#pragma once
#include "vector"
#include "time.h"
using namespace std;

#define SIGMOID 1
#define SOFTMAX 2
#define TANH 3
#define SE 4 //square error

//node type
//include input weight,net value and output value
struct Perceptron {
	int inputWeightNum;//该节点的输入个数
	double* inputWeights;//输入权值数组
	double netValue,output,delta;//结点值，节点输出值，该节点之后网络的损失值之和
	Perceptron() {

	}
	Perceptron(int n)
	{
		double x,sum=0;
		inputWeightNum = n;
		inputWeights = new double[n];
		for (int i = 0; i < n; i++) {
			x = (double)rand() / (double)RAND_MAX;//随机初始化，范围在[0,0.5] 
			inputWeights[i] = (x);
			sum += x;
		}
		//权值归一化
		for (int i = 0; i < n; i++) {
			inputWeights[i] /= sum;
		}
	}
	~Perceptron(){

	}
};

//layer type
//include: a group of node
struct Layer {
	int perceptronNum; //该层节点个数
	Perceptron *perceptrons; //结点数组
	Layer() {

	}
	//初始化层 参数：结点数，前一层节点数，是否加偏置
	Layer(int n,int perN,bool partialNum)
	{
		perceptronNum = n;
		srand((unsigned)time(NULL));
		perceptrons = new Perceptron[n+1];
		int i;
		//每个节点的具体信息
		for (i = 0; i < n; i++) {
			Perceptron perceptron(perN);
			perceptrons[i] = perceptron;
		}
		//加偏置
		if (partialNum) {
			perceptronNum++;
			Perceptron perceptron(0);
			perceptron.netValue = 1;
			perceptron.output = 1;
			perceptrons[i] = perceptron;
		}
	}
	~Layer()
	{

	}
};


//a sample data
struct Sample{
	int featureNum; //特征向量大小
	double* feature;//特征向量
	double* label;//标签向量
	Sample() {

	}
	//初始化样本 参数：特征向量大小
	Sample(int n) {
		featureNum = n;
		feature = new double[featureNum];
		label = new double[featureNum];
	}
	//重载赋值运算符
	struct Sample& operator=(const struct Sample& s) {
		this->featureNum = s.featureNum;
		for (int i = 0; i < featureNum; i++) {
			this->feature[i] = s.feature[i];
			this->label[i] = s.label[i];
		}
		return *this;
	}
};

struct Data {
	int sampleNum; //数据集中样本个数
	struct Sample *sample;//样本集
	struct Data() {
		sampleNum = 0;
	}
	//初始化数据集 参数：样本数，特征数
	struct Data(int sampleNum,int featureNum) {
		this->sampleNum = sampleNum;
		this->sample = new struct Sample[sampleNum];
		for (int i = 0; i < sampleNum; ++i) {
			this->sample[i].feature = new double[featureNum];
			this->sample[i].label = new double[featureNum];
		}
	}
};

//ANN type
//include: a group of layer
class ANN
{
public:

	// 结构相关
	ANN();
	//新建ANN，设置参数(输入层数，隐层数，输出层数，以及各层的结点个数)
	ANN(vector<int>& layer);
	~ANN();
	void setParameter(double learningRate, int step,double e) { //设定参数
		this->learningRate = learningRate;
		this->step = step;
		this->e = e;
	}
	int getNumLayer() {
		return this->hiddenLayerNum + 2;
	}
	struct Layer* getLayers() { return layers; }//获取层参数信息
	void setLearningRate(double learningRate) {
		this->learningRate = learningRate;
	}

	// 前向传播
	double activeFunction(double netValue, int func); // 求经过激活函数的结果
	double weightedSum(int layerNum,int perceptronNum); // 加权和
	void forward(); //前向传播
	double loss(int func);
	double square_error();//计算对应节点均方误差
	double getSquareError() { //获取均方误差
		return this->curSquareError;
	}

	// 后向传播
	void backward(); //反向传播
	double activeFunctionD(double netValue, int func); // 求经过激活函数的导数
	double Loss_functionD(int func,double target,double out);//损失函数的导数
	void updateWeights(int layerNum); //更新权值
	double computeA(int layerNum, int perceptronNum);//计算此节点为总误差的贡献率
	
	void train();

	// 预测
	double* prediction(struct Sample& sample);
	int judgeClassification(double* v);

	// 数据
	void addData(int num);//载入数据
	void addTestData(int num);//载入测试数据

	// 参数存储 、 加载
	bool saveANN(string fileName);
	bool loadANN(string fileName);

	// 测试
	double test();
private:
	struct Layer* layers;//input hidden（每个隐层自动加一个偏置） output 
	int hiddenLayerNum;// the number of hidden layer

	double learningRate;//learning rate
	int step;//迭代步数
	double curSquareError;//当前误差率
	double e;//精度

	struct Sample* sample;//sample
	struct Data* data;//data
	struct Data* testData;//data

};



/*
收敛：若干次后误差不在变
收敛到最优解还是次优解
卷积核：多尺度提取特征
DL
*/

#endif // !ANN_H