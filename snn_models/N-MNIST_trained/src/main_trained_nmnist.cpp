/* The MIT License (MIT)

Copyright (c) 2016 Cognitive Anteater Robotics Laboratory @ University of Calfornia, Irvine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

* CARLsim
* created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
* maintained by:
* (MA) Mike Avery <averym@uci.edu>
* (MB) Michael Beyeler <mbeyeler@uci.edu>,
* (KDC) Kristofor Carlson <kdcarlso@uci.edu>
* (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/

/* 
Copyright (c) 2021 Electronics and Telecommunications Research Institute

This is a pre-trained SNN model trained using the MM-BP algorithm, 
and the model run on the CARLsim4 SNN simulator.

created and maintained by:
Eunji Pak <pakeunji@etri.re.kr>
YoungMok Ha <ymha@etri.re.kr>
*/

#include <carlsim.h>
#include <spikegen_from_vector.h>

#include <stdlib.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#define nTotalTestData 10

int nTestData = 10;
float wtScale = 1.0f;

struct __spike_train {
	int label;
	std::vector< std::vector<int> > *strains;
};

struct __spike_train spike_train[nTotalTestData];

struct weight {
	int src;
	int dst;
	float weight;
};
void readTrainedWeights(char* filename, struct weight *weights, int Len);
bool getFileContentByLine(std::string fileName, int sLine, int eLine, std::vector<float> & vecfloat);
void readNMNISTData();

int total_num_spikes = 0;
int layer_num_spikes[5];

class customSpikes: public SpikeGenerator{ 
	public:
		customSpikes() {
			this->startTime = 0;
		}

		std::vector <std::vector <int> > *spktrains;
		std::vector<int> *cindex;
		int startTime;

		int setSpikeTrains(std::vector <std::vector <int> > *spktrains, int startTime, int nsize) {
			this->cindex = new std::vector<int>(nsize, 0);
			this->spktrains = spktrains; 
			this->startTime = startTime;

			return 0;
		}

		int nextSpikeTime(CARLsim* sim, int grpId, int nid, 
				int currentTime, int lastScheduledSpikeTime, int endOfTimeSlice) {

			std::vector<int> times = (*this->spktrains)[nid];
			int currentIndex_ = (*cindex)[nid];

			int size_ = (*this->spktrains)[nid].size();
			if((currentIndex_ < size_) && (times[currentIndex_] < endOfTimeSlice)) {
				int t = times[currentIndex_] + startTime;
				(*cindex)[nid]++;

				return t;
			}
			return -1;
		}
};


// connection to implement lateral inhibition
class connectWTA: public ConnectionGenerator {
	public:
		connectWTA() {}
		~connectWTA() {}

		void connect(CARLsim* sim, int srcGrp, int i, int destGrp, int j, 
				float& weight, float &maxWt, float &delay, bool& connected) {
			maxWt = 50.0f * wtScale;
			delay = 1;
			connected = (i != j);
			weight = wtScale * 1.0f; // -1.0f weight for lateral inhibition 
		}
};


class connectLtoL: public ConnectionGenerator {
	public:
		connectLtoL(struct weight *trainedWts, bool usePosWts) {
			this->usePosWts = usePosWts; 
			this->trainedWts = trainedWts;
		}
		~connectLtoL() {}

		bool usePosWts; 
		struct weight *trainedWts;

		void connect(CARLsim* sim, int srcGrp, int i, int destGrp, int j, 
				float& weight, float &maxWt, float &delay, bool& connected) {
			maxWt = 50.0f * wtScale;
			delay = 1;
			connected = false; // init
			weight = 0.0f; // init

			int X = sim->getGroupNumNeurons(srcGrp);
			int Y = sim->getGroupNumNeurons(destGrp);
			int index = j * X + i;

			if(usePosWts) { // use only the positive weights 
                connected = (trainedWts[index].weight > 0);
				weight = wtScale * trainedWts[index].weight;
			}
			else { // use only the negative weights
				connected = (trainedWts[index].weight < 0);
				weight = wtScale * -(trainedWts[index].weight);
			}
		}
};

#define nIn 2312
#define nHid 800
#define nOut 10

struct st_spike_info {
	int neuron_id;
	int num_spikes;
	int layer_id;

	float sum_incoming_weights;
	float sum_outgoing_weights;
};
struct st_spike_info spike_info[(nIn + nIn + nHid + nHid + nOut)];
int spike_info_index = 0;
std::vector< std::vector<int> > *input_spikes;

int main(int argc, const char* argv[]) {
	// ---------------- CONFIG STATE -------------------
	// create a network on GPU
	CARLsim sim("mmbp-trained-SNN-N-MNIST", GPU_MODE, USER);

	float vth_hid = 15.0f;
	float vth_out = 15.0f;
	float vr_hid = 0.0f;
	float vr_out = 0.0f;
	float rmem = 1.0f;
	float ampa = 8.0f;
	float gabaa = 8.0f;
	float nmda = 1.0f;
	float gabab = 1.0f;
	float tau_m = 64.0f;
	float tau_ref = 2.0f;

	int run = 500;
	int rest = 500;
	int inter = run + rest; 

	// load trained weights 
	struct weight *wtsIntoHid;
	struct weight *wtsHidtoOut;

	int size_weight = sizeof(struct weight); 
	wtsIntoHid = (struct weight *)malloc(size_weight * nIn * nHid);
	wtsHidtoOut = (struct weight *)malloc(size_weight * nHid * nOut);

	if((wtsIntoHid == NULL) || (wtsHidtoOut == NULL)) return -1;

	for(int i = 0; i < (size_weight * nIn * nHid); i++) {
		wtsIntoHid->src = 0; wtsIntoHid->dst = 0; wtsIntoHid->weight = 0.0f;
	}
	for(int i = 0; i < (size_weight * nHid * nOut); i++) {
		wtsHidtoOut->src = 0; wtsHidtoOut->dst = 0; wtsHidtoOut->weight = 0.0f;
	}

	readTrainedWeights((char*)"weight/weight_L1L2.txt", wtsIntoHid, nIn * nHid);
    readTrainedWeights((char*)"weight/weight_L2L3.txt", wtsHidtoOut, nHid * nOut);

	// load NMNIST data and label to spike_train struct
	input_spikes = new std::vector< std::vector<int> >(2312, std::vector<int>());
	readNMNISTData();
	// 1.configure the network
	/* (note) Inhibitory neurons are used for the connection with negative weight value or
	   Lateral inhibition of output layer  */
	// Create Input Layer 
	int gInExc = sim.createSpikeGeneratorGroup("input-exc", nIn, EXCITATORY_NEURON);
	int gInInh = sim.createSpikeGeneratorGroup("input-inh", nIn, INHIBITORY_NEURON);
	PoissonRate silent(nIn);

	std::vector< std::vector<int> > *spikes;
	spikes = new std::vector< std::vector<int> >(nIn, std::vector<int>());

	std::vector< std::vector<int> > *silent_spikes;
	silent_spikes = new std::vector< std::vector<int> >(nIn, std::vector<int>());

	for(int i = 0; i < silent_spikes->size(); i++) { 
		for(int j = 0; j < (*silent_spikes)[i].size(); j++) {
			(*silent_spikes)[i][j] = 0;
		}
	}

	// setup input silent_spikes
	customSpikes in1_spikes;
	customSpikes in2_spikes;
	sim.setSpikeGenerator(gInExc, &in1_spikes);
	sim.setSpikeGenerator(gInInh, &in2_spikes);

	/*
	   spikes = spike_train[1].strains;
	   in1_spikes.setSpikeTrains(spikes);
	 */

	// Create Hidden and Output Layer
	int gHidExc = sim.createGroupMMBP("hidden-ext", nHid, EXCITATORY_NEURON);
	int gHidInh = sim.createGroupMMBP("hidden-inh", nHid, INHIBITORY_NEURON);
	int gOutInh = sim.createGroupMMBP("output-inh", nOut, INHIBITORY_NEURON);

	// setNeuromParametersMMBP(neuron_group_ID, tau_m, tau_ref, vTh, vReset, RangeRmem(rMem));

	sim.setNeuronParametersMMBP(gHidExc, tau_m, tau_ref, (vth_hid), (vr_hid), RangeRmem(rmem));
	sim.setNeuronParametersMMBP(gHidInh, tau_m, tau_ref, (vth_hid), (vr_hid), RangeRmem(rmem));
	sim.setNeuronParametersMMBP(gOutInh, tau_m, tau_ref, (vth_out), (vr_out), RangeRmem(rmem));

	sim.connect(gInExc, gHidExc, new connectLtoL(wtsIntoHid, true), SYN_FIXED);
	sim.connect(gInExc, gHidInh, new connectLtoL(wtsIntoHid, true), SYN_FIXED);
	sim.connect(gInInh, gHidExc, new connectLtoL(wtsIntoHid, false), SYN_FIXED);
	sim.connect(gInInh, gHidInh, new connectLtoL(wtsIntoHid, false), SYN_FIXED);
	sim.connect(gHidExc, gOutInh, new connectLtoL(wtsHidtoOut, true), SYN_FIXED);
	sim.connect(gHidInh, gOutInh, new connectLtoL(wtsHidtoOut, false), SYN_FIXED);	
	sim.connect(gOutInh, gOutInh, new connectWTA(), SYN_FIXED);

	sim.setConductances(true, ampa, nmda, gabaa, gabab);

	// ---------------- SETUP STATE -------------------
	// 2.build the network

	SpikeMonitor *sm_HidExc, *sm_HidInh, *sm_InExc, *sm_InInh, *sm_OutInh;
	ConnectionMonitor *cm_InExc_HidExc, *cm_InExc_HidInh, *cm_InInh_HidExc, *cm_InInh_HidInh, 
					  *cm_HidExc_OutInh, *cm_HidInh_OutInh;
	/*
	   std::vector< std::vector<float> > wt = cm_HidExc_OutInh->takeSnapshot();
	   printf("w[1][197] = %f\n", wt[197][1]);
	 */

	sm_InExc = sim.setSpikeMonitor(gInExc, "DEFAULT");
	sm_InInh = sim.setSpikeMonitor(gInInh, "DEFAULT");
	sm_HidExc = sim.setSpikeMonitor(gHidExc, "DEFAULT");
	sm_HidInh = sim.setSpikeMonitor(gHidInh, "DEFAULT");
	sm_OutInh = sim.setSpikeMonitor(gOutInh, "DEFAULT");

	sim.setupNetwork();
	//	int inter = 2000;
	cm_InExc_HidExc = sim.setConnectionMonitor(gInExc, gHidExc, "DEFAULT");
	cm_InExc_HidInh = sim.setConnectionMonitor(gInExc, gHidInh, "DEFAULT");
	cm_InInh_HidExc = sim.setConnectionMonitor(gInInh, gHidExc, "DEFAULT");
	cm_InInh_HidInh = sim.setConnectionMonitor(gInInh, gHidInh, "DEFAULT");
	cm_HidExc_OutInh = sim.setConnectionMonitor(gHidExc, gOutInh, "DEFAULT");
	cm_HidInh_OutInh = sim.setConnectionMonitor(gHidInh, gOutInh, "DEFAULT");


	std::vector<int> rand_index;
	for(int i = 0; i < nTotalTestData; i++) { rand_index.push_back(i); }
	random_shuffle(rand_index.begin(), rand_index.end());

	int rindex = 0;
	int right = 0;
	int wrong = 0;
	for(int k = 0; k < nTestData; k++) {
		rindex = rand_index[k];

		in1_spikes.setSpikeTrains(silent_spikes, inter * k, nIn);
		in2_spikes.setSpikeTrains(silent_spikes, inter * k, nIn);
		sim.runNetwork(rest/1000, rest%1000);

		spikes = spike_train[rindex].strains;
		int label = spike_train[rindex].label; 

		in1_spikes.setSpikeTrains(spikes, inter * k + rest, nIn);
		in2_spikes.setSpikeTrains(spikes, inter * k + rest, nIn);

		sm_InExc->startRecording();
		sm_InInh->startRecording();
		sm_HidExc->startRecording();
		sm_HidInh->startRecording();
		sm_OutInh->startRecording();

		sim.runNetwork(run/1000, run%1000);
		sm_InExc->stopRecording();
		sm_InInh->stopRecording();
		sm_OutInh->stopRecording();
		sm_HidExc->stopRecording();
		sm_HidInh->stopRecording();

		int spikes_out[nOut];
		int maxS = -1, winner = -1;
		for(int nk = 0; nk < nOut; nk++) {
			spikes_out[nk] = sm_OutInh->getNeuronNumSpikes(nk);
			if(maxS < spikes_out[nk]) { maxS = spikes_out[nk]; winner = nk;}
		}

		if(winner == label) right++;
		else wrong++;
		printf("%dth data, [PREDICTED | ANSWER] = [%d | %d], %d\n", rindex, winner, label, winner == label);

	}


	printf("accuracy = %.4f\n", (float)(right)/(float)(right + wrong) * 100.0f);
	free(spikes);
	return 0;
}

void readTrainedWeights(char* filename, struct weight *weights, int Len) {
	FILE* file;
	file = fopen(filename, "r");

	for(int i = 0; i < Len; i++) {
		fscanf(file, "%d\t%d\t%f\n", &(weights[i].src), &(weights[i].dst), &(weights[i].weight));
	}
	fclose(file);
}

void readNMNISTData() {
	std::ifstream fin("sample_input/input.dat");
	std::string times;

	bool change_data = false;

	for(int index = 0; index < nTotalTestData; index++) {
		while(getline(fin, times)) {
			if(times[0] == '#') {
				spike_train[index].strains = input_spikes;
				spike_train[index].label = (times[2]-'0');
				change_data = true;
			}
			if(change_data) { change_data = false; break;}

			std::istringstream iss(times);
			int neuronindex = -1;
			iss>>neuronindex;
			neuronindex = neuronindex - 1;

			int tspike;
			while(iss >> tspike) { (*input_spikes)[neuronindex].push_back(tspike); }
		}
		if(change_data) continue;
	}
}

