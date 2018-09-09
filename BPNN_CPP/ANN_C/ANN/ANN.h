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
	int inputWeightNum;//�ýڵ���������
	double* inputWeights;//����Ȩֵ����
	double netValue,output,delta;//���ֵ���ڵ����ֵ���ýڵ�֮���������ʧֵ֮��
	Perceptron() {

	}
	Perceptron(int n)
	{
		double x,sum=0;
		inputWeightNum = n;
		inputWeights = new double[n];
		for (int i = 0; i < n; i++) {
			x = (double)rand() / (double)RAND_MAX;//�����ʼ������Χ��[0,0.5] 
			inputWeights[i] = (x);
			sum += x;
		}
		//Ȩֵ��һ��
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
	int perceptronNum; //�ò�ڵ����
	Perceptron *perceptrons; //�������
	Layer() {

	}
	//��ʼ���� �������������ǰһ��ڵ������Ƿ��ƫ��
	Layer(int n,int perN,bool partialNum)
	{
		perceptronNum = n;
		srand((unsigned)time(NULL));
		perceptrons = new Perceptron[n+1];
		int i;
		//ÿ���ڵ�ľ�����Ϣ
		for (i = 0; i < n; i++) {
			Perceptron perceptron(perN);
			perceptrons[i] = perceptron;
		}
		//��ƫ��
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
	int featureNum; //����������С
	double* feature;//��������
	double* label;//��ǩ����
	Sample() {

	}
	//��ʼ������ ����������������С
	Sample(int n) {
		featureNum = n;
		feature = new double[featureNum];
		label = new double[featureNum];
	}
	//���ظ�ֵ�����
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
	int sampleNum; //���ݼ�����������
	struct Sample *sample;//������
	struct Data() {
		sampleNum = 0;
	}
	//��ʼ�����ݼ� ��������������������
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

	// �ṹ���
	ANN();
	//�½�ANN�����ò���(���������������������������Լ�����Ľ�����)
	ANN(vector<int>& layer);
	~ANN();
	void setParameter(double learningRate, int step,double e) { //�趨����
		this->learningRate = learningRate;
		this->step = step;
		this->e = e;
	}
	int getNumLayer() {
		return this->hiddenLayerNum + 2;
	}
	struct Layer* getLayers() { return layers; }//��ȡ�������Ϣ
	void setLearningRate(double learningRate) {
		this->learningRate = learningRate;
	}

	// ǰ�򴫲�
	double activeFunction(double netValue, int func); // �󾭹�������Ľ��
	double weightedSum(int layerNum,int perceptronNum); // ��Ȩ��
	void forward(); //ǰ�򴫲�
	double loss(int func);
	double square_error();//�����Ӧ�ڵ�������
	double getSquareError() { //��ȡ�������
		return this->curSquareError;
	}

	// ���򴫲�
	void backward(); //���򴫲�
	double activeFunctionD(double netValue, int func); // �󾭹�������ĵ���
	double Loss_functionD(int func,double target,double out);//��ʧ�����ĵ���
	void updateWeights(int layerNum); //����Ȩֵ
	double computeA(int layerNum, int perceptronNum);//����˽ڵ�Ϊ�����Ĺ�����
	
	void train();

	// Ԥ��
	double* prediction(struct Sample& sample);
	int judgeClassification(double* v);

	// ����
	void addData(int num);//��������
	void addTestData(int num);//�����������

	// �����洢 �� ����
	bool saveANN(string fileName);
	bool loadANN(string fileName);

	// ����
	double test();
private:
	struct Layer* layers;//input hidden��ÿ�������Զ���һ��ƫ�ã� output 
	int hiddenLayerNum;// the number of hidden layer

	double learningRate;//learning rate
	int step;//��������
	double curSquareError;//��ǰ�����
	double e;//����

	struct Sample* sample;//sample
	struct Data* data;//data
	struct Data* testData;//data

};



/*
���������ɴκ����ڱ�
���������Ž⻹�Ǵ��Ž�
����ˣ���߶���ȡ����
DL
*/

#endif // !ANN_H