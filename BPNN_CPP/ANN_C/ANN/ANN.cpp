#include "stdafx.h"
#include "ANN.h"
#include "cmath"
#include "iostream"
#include "fstream"
#include "vector"
#include "sstream"
#include "cstdio"
using namespace std;
ANN::ANN()
{
}

// 新建ANN，设置参数(输入层数，隐层数，输出层数，以及各层的结点个数)
ANN::ANN(vector<int>& layer)
{
	int len = layer.size();
	this->hiddenLayerNum = len-2;	
	this->layers = new struct Layer[len];
	//新建每一层节点
	struct Layer tmpLayer(layer[0], 0,true);
	this->layers[0] = tmpLayer;
	bool f = true;
	for(int i=1;i<len;i++){
		if (i == len - 1) {
			f = false;
		}
		struct Layer tmpLayer(layer[i],this->layers[i-1].perceptronNum,f);
		this->layers[i] = (tmpLayer);
	}
}
	
ANN::~ANN()
{

}

// 计算加权和
double ANN::weightedSum(int layerNum,int perceptronNum)
{
	double res = 0.0;
	if(layerNum<=0 || perceptronNum<0) {
		return res;
	}
	int inputNum = this->layers[layerNum].perceptrons[perceptronNum].inputWeightNum;//本节点输入的个数
	for(int i=0;i<inputNum;i++)
	{
		//前一层每个节点的输出值*此节点对应输入的权值
		res += this->layers[layerNum -1].perceptrons[i].output*this->layers[layerNum].perceptrons[perceptronNum].inputWeights[i];
	}
	return res;
}

//计算经过激活函数后结果
double ANN::activeFunction(double netValue,int func)
{
	if(func == SIGMOID){
		return  1.0/(1+exp(-netValue));
	}
	if (func == SOFTMAX) {
		return 0;
	}
	if (func == TANH) {
		return (exp(netValue) - exp(-netValue)) / (exp(netValue) + exp(-netValue));
	}
	return netValue;
}

// 求经过激活函数的导数
double ANN::activeFunctionD(double netValue, int func) { 
	if (func == SIGMOID) {
		return  netValue*(1- netValue);
	}
	if (func == SOFTMAX) {
		return 0;
	}
	if (func == TANH) {
		return 1- netValue* netValue;
	}
	return netValue;
}

// 损失函数的导数
double ANN::Loss_functionD(int func, double target, double out)
{
	if (func == SE) {
		return -(target - out);
	}
	return out;
}

// 为每个结点计算传输值
void ANN::forward()
{
	int totalNum = this->hiddenLayerNum + 2,pnum=0;
	//input layer (输入数据为矩阵，一行为一个样本即特征向量，列数表示特征数，行数代表训练样本数) 输入层输出的值即读入的值
	pnum = layers[0].perceptronNum - 1;
	for(int i=0;i<pnum;i++){
		layers[0].perceptrons[i].netValue = this->sample->feature[i];
		layers[0].perceptrons[i].output = layers[0].perceptrons[i].netValue;
	}			
	//hidden and output layer 
	for(int i=1;i<totalNum;i++){
		pnum = layers[i].perceptronNum;
		if (i != totalNum - 1) {
			pnum--;
		}
		//计算 第 i layer 的每个节点节点值与输出值
		for (int j = 0; j < pnum; j++) {
			layers[i].perceptrons[j].netValue = weightedSum(i,j);
			layers[i].perceptrons[j].output = activeFunction(layers[i].perceptrons[j].netValue,SIGMOID);
		}
	}
	square_error();
}

// 均方误差
double ANN::square_error()
{
	double E_total = 0;
	int outputLayerId = this->hiddenLayerNum+1;
	int outNum =layers[outputLayerId].perceptronNum;
	double* E = new double[outNum];
	for(int i=0;i<outNum;i++)
	{
		E[i] = 0.5*pow((this->sample->label[i] - this->layers[outputLayerId].perceptrons[i].output),2);
		E_total +=E[i];
	}
	this->curSquareError = E_total;
	return E_total;
}

double ANN::loss(int func)
{
	if (func == SE) {
		return square_error();
	}
	else if (func == SOFTMAX) {

	}
}
// 计算某结点贡献的误差
double ANN::computeA(int layerNum, int perceptronNum)
{
	this->layers[layerNum].perceptrons[perceptronNum].delta = 0;
	int pointNum = this->layers[layerNum + 1].perceptronNum;
	//某节点之前的误差用delta表示，等于上一层每个结点的delta与对应权值乘积之和 再乘上本节点的损失
	for (int i = 0; i < pointNum; i++) {
		this->layers[layerNum].perceptrons[perceptronNum].delta
			+= this->layers[layerNum + 1].perceptrons[i].delta 
			* this->layers[layerNum + 1].perceptrons[i].inputWeights[perceptronNum];
	}
	this->layers[layerNum].perceptrons[perceptronNum].delta
		*= activeFunctionD(this->layers[layerNum].perceptrons[perceptronNum].output, SIGMOID);
	return 0;
}


// 权值更新
void ANN::updateWeights(int layerNum){
	
	double t_o, o_n, n_w,t_w;
	//隐含层---->输出层的权值更新
	if(layerNum == this->hiddenLayerNum+1){
		int numP= this->layers[layerNum].perceptronNum,numW;
		for (int i = 0; i < numP; i++) {
			//总误差对输出层节点偏导 E_total / out01
			t_o = Loss_functionD(SE,this->sample->label[i], 
								this->layers[layerNum].perceptrons[i].output);
			
			//输出层输出值对求输出层节点值求偏导 out01 / net01
			o_n = activeFunctionD(this->layers[layerNum].perceptrons[i].output,
									SIGMOID);

			this->layers[layerNum].perceptrons[i].delta = t_o * o_n;
			//对outi的所有权值进行修正
			numW = this->layers[layerNum].perceptrons[i].inputWeightNum;
			for (int j = 0; j < numW; j++) {
				//节点值对权值求偏导 net01 / w
				n_w = this->layers[layerNum-1].perceptrons[j].output;
				//三者相乘 即整体误差对权值的偏导值 X
				t_w = this->layers[layerNum].perceptrons[i].delta * n_w;
				//更新权重值 w = w -a*X
				this->layers[layerNum].perceptrons[i].inputWeights[j] 
					-= this->learningRate * t_w;
			}
		}
	}
	//隐含层---->隐含层的权值更新 out(h1) net(h1) w1
	else if(layerNum >= 1 && layerNum < this->hiddenLayerNum + 1){
		int numP = this->layers[layerNum].perceptronNum, numW;
		for (int i = 0; i < numP; i++) {
			//总误差对输出层节点偏导 E_total / out(h1) = E01 / out(h1) + E02 / out(h1) 误差相加
			computeA(layerNum, i);

			//对hidden i的所有权值进行修正
			numW = this->layers[layerNum].perceptrons[i].inputWeightNum;
			for (int j = 0; j < numW; j++) {
				//节点值对权值求偏导 net(h1) / w
				n_w = this->layers[layerNum-1].perceptrons[j].output;
				//三者相乘 即整体误差对权值的偏导值 X
				t_w = this->layers[layerNum].perceptrons[i].delta * n_w;
				//更新权重值 w = w -a*X
				this->layers[layerNum].perceptrons[i].inputWeights[j] -= this->learningRate * t_w;
			}
		}
	}
	else {
		std::cerr << "update parameter wrong \n";
		return;
	}
}


// 反馈过程
void ANN::backward()
{
	
	for (int i = this->hiddenLayerNum + 1; i >= 1; i--) {
		updateWeights(i);
	}
}


// 训练过程
void ANN::train()
{
	int sampleNum = (*this->data).sampleNum;
	int con = 0;
	this->sample = new struct Sample(this->data->sample->featureNum);
	for (int i =0; i<sampleNum;i++) {
		*this->sample = (*this->data).sample[i];
		//前向传播 反向传播交替
		forward();
		backward();
	}
}

// 输入一样本，输出其结果向量
double* ANN::prediction(struct Sample& sample)
{
	double* res = new double[sample.featureNum];
	*this->sample = sample;
	forward();
	int numO = this->layers[this->hiddenLayerNum + 1].perceptronNum;
	for (int j = 0; j < numO; j++) {
		res[j] = (this->layers[this->hiddenLayerNum + 1].perceptrons[j].output);
	}
	return res;
}

// 根据结果向量判断类别
int ANN::judgeClassification(double* v)
{
	double Max = INT_MIN;
	int type = -1;
	int numO = this->layers[this->hiddenLayerNum + 1].perceptronNum;
	for (int j = 0; j < numO; j++) {
		if (Max < this->layers[this->hiddenLayerNum + 1].perceptrons[j].output) {
			Max = this->layers[this->hiddenLayerNum + 1].perceptrons[j].output;
			type = j;
		}
	}
	return type;
}

// 保存ANN 参数
bool ANN::saveANN(string fileName)
{
	ofstream out(fileName);
	if (out.is_open())
	{
		std::cout << "file write begin !!\n";
		out << this->hiddenLayerNum << endl;
		out << this->learningRate << endl;
		out << this->step << endl;
		out << this->curSquareError << endl;
		int numL = hiddenLayerNum + 2,numP,numI;
		for (int i = 0; i < numL; i++) {
			numP = this->layers[i].perceptronNum;
			out << numP << " ";
		}
		out << endl;
		for (int i = 1; i < numL; i++) {
			numP = this->layers[i].perceptronNum;
			for (int j = 0; j < numP; j++) {
				numI = this->layers[i].perceptrons[j].inputWeightNum;
				for (int k = 0; k < numI; k++) {
					out << this->layers[i].perceptrons[j].inputWeights[k] << "   ";
				}
				out << endl;
			}
			out << endl << endl;
		}
		out.close();
		std::cout << "file write end!!\n";
		return true;
	}
	else {
		return false;
	}
}

// 加载ANN参数
bool ANN::loadANN(string fileName)
{
	ifstream fin(fileName);
	if (!fin.is_open())
	{
		cerr << "file open wrong\n";
		return false;
	}
	fin >> this->hiddenLayerNum >> this->learningRate >> this->step >> this->curSquareError;
	int numL = hiddenLayerNum + 2, numP, numI;
	vector<int> layerD;
	for (int i = 0; i < this->hiddenLayerNum + 2; i++) {
		fin >> numP;
		layerD.push_back(numP);
	}
	//ANN res(layerD);
	for (int i = 1; i < numL; i++) {
		numP = this->layers[i].perceptronNum;
		for (int j = 0; j < numP; j++) {
			numI = this->layers[i].perceptrons[j].inputWeightNum;
			for (int k = 0; k < numI; k++) {
				fin >> this->layers[i].perceptrons[j].inputWeights[k];
			}
		}
	}
	fin.close();
	return true;
}
//==================================== DATA ============================================

//加载训练数据 / 批次
void ANN::addData(int num)
{
	delete (this->data);
	stringstream ss;
	ss << num;
	string str = ss.str();
	int r, c, x, labelNumber;
	ifstream fin("C:/Project/opencv/Mnist-txt/train_batch_60_random/train/" + str + ".txt", std::ifstream::in);
	if (fin.is_open()) {
		fin >> r >> c;
		c--;
		this->data = new struct Data(r,c);
		struct Sample* sample = new Sample(c);
		for (int i = 0; i < r; ++i) {
			fin >> labelNumber;
			for (int j = 0; j < 10; j++) {
				if (j == labelNumber) {
					sample->label[j] = 1;
				}
				else {
					sample->label[j] = 0;
				}
			}
			for (int j = 0; j < c; ++j)
			{
				fin >> x;
				sample->feature[j] = (x);
			}
			(*this->data).sample[i] = *sample;
			//if(i >= 1)
				//cout << (*this->data).sample[i - 1].feature[10]<<" "<<(*this->data).sample[i].feature[10] <<"-*********----" << endl;
		}
		fin.close();
	}
}

// 加载测试数据
void ANN::addTestData(int num)
{
	delete (this->testData);
	stringstream ss;
	ss << num;
	string str = ss.str();
	int r, c, x;
	ifstream fin("C:/Project/opencv/Mnist-txt/test_batch/" + str + ".txt", std::ifstream::in);
	if (fin.is_open()) {
		fin >> r >> c;
		this->testData = new struct Data(r,c);
		struct Sample* sample = new Sample(c);
		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j)
			{
				fin >> x;
				sample->feature[j] = (x);
			}
			(*this->testData).sample[i] = *(sample);
		}
		fin.close();
	}
	double* tmp = new double[c];
	for (int j = 0; j < 10; j++) {
		tmp[j] = 0;
	}
	tmp[num] = 1;
	r = (*this->testData).sampleNum;
	for (int i = 0; i < r; ++i) {
		*(*this->testData).sample[i].label = *tmp;
	}

	std::cout << "Adding test data " << num << " is over\n";
}

//加载训练数据 / 批次
/*
void ANN::addData(int num)
{
	stringstream ss;
	ss << num;
	string str = ss.str();
	int r, c, x;
	ifstream fin("C:/Project/opencv/Mnist-txt/train_batch_60/train/"+str+".txt", std::ifstream::in);
	if (fin.is_open()) {
		fin >> r >> c;
		this->data = new struct Data(r);
		struct Sample sample(c);
		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j)
			{
				fin >> x;
				sample.feature[j] = x;
			}
			(*this->data).sample[i] = (sample);
		}
		fin.close();
	}
	ifstream fin1("C:/Project/opencv/Mnist-txt/train_batch_60/label/"+str+".txt", std::ifstream::in);
	if (fin1.is_open()) {
		fin1 >> r >> c;
		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j)
			{
				fin1 >> x;
				(*this->data).sample[i].label[j] = (x);
			}
		}
		fin1.close();
	}
	std::cout << "Adding data "<<num<<" is over\n";
}
*/
//==================================== TEST ============================================
double ANN::test()
{
	int type;
	double *tmp;
	int conWrong = 0,numTotal = 0;
	//10 批test
	for (int i = 0; i < 10; i++) {
		addTestData(i); 
		int numS = (*this->testData).sampleNum;
		numTotal += numS;
		//每个样本判断类别
		for (int j = 0; j < numS; j++) {
			tmp = prediction((*this->testData).sample[j]);
			type =this->judgeClassification(tmp);
			if (type != i) {
				conWrong++;
			}
		}
	}
	
	cout << numTotal << " " << conWrong << endl;
	return 100.0*(double)(numTotal - conWrong) / (double)numTotal;
}

