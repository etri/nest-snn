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
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int nTotalTestData = 10000;
int nTestData = 10000;

//typedef unsigned long uint32_t;
static uint32_t swapU32(uint32_t value);
static void readDataHeader(std::ifstream& file, int &n_imgs, int &n_rows, int &n_cols);
static void readLabelHeader(std::ifstream& file, int &n_imgs);
static void readData(std::vector< std::vector<int> > &intensity);
static void readLabel(std::vector<int> &label);

float wtScale = 1.0f;

struct weight {
    int src;
    int dst;
    float weight;
};

void readTrainedWeights(char* filename, struct weight *weights, int Len);
//bool getFileContentByLine(std::string fileName, int sLine, int eLine, std::vector<float> & vecfloat);

// connection to implement lateral inhibition
class connectWTA: public ConnectionGenerator {
    public:
        connectWTA() {}
        ~connectWTA() {}

        void connect(CARLsim* sim, int srcGrp, int i, int destGrp, int j, 
                float& weight, float &maxWt, float &delay, bool& connected) {
            maxWt = 10.0f * wtScale;
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
            maxWt = 10.0f * wtScale;
            delay = 1;
            connected = false; // init
            weight = 0.0f; // init

            int X = sim->getGroupNumNeurons(srcGrp);
            int Y = sim->getGroupNumNeurons(destGrp);
            int index = j * X + i;

            //			printf("w[%d][%d] = %f\n", i, j, trainedWts[index].weight);

            if(usePosWts) { // use only the positive weights 
                connected = (trainedWts[index].weight > 0);
//                weight = round_weight(trainedWts[index].weight);
                weight = trainedWts[index].weight;
            }
            else { // use only the negative weights
                connected = (trainedWts[index].weight < 0);
//                weight = -round_weight(trainedWts[index].weight);
                weight = -(trainedWts[index].weight);
            }
        }
};

int main(int argc, const char* argv[]) {
    // ---------------- CONFIG STATE -------------------
    // create a network on GPU
    CARLsim sim("mmbp-trained-SNN-MNIST", GPU_MODE, USER);

    float tau_m = 64.0f;
    float tau_ref = 2.0f;
    float vth_hid = 20.0f;
    float vth_out = 8.0f;
    float vr_hid = 0.0f;
    float vr_out = 0.0f;
    float rmem = 1.0f;
    int rest = 200;
    int run = 200;
    float ampa = 8.0f;
    float gabaa = 8.0f;

    float nmda = 8.0f;
    float gabab = 8.0f;

    // read MNIST data (data and label)
    std::vector< std::vector<int> > intensity;
    std::vector<int> label;

    readData(intensity); readLabel(label);

    int nIn = 784;
    int nHid = 800;
    int nOut = 10;

    // load trained weights 

    struct weight wtsIntoHid[nIn * nHid];
    struct weight wtsHidtoOut[nHid * nOut];

	readTrainedWeights((char*)"weight/weight_L1L2.txt", wtsIntoHid, nIn * nHid);
    readTrainedWeights((char*)"weight/weight_L2L3.txt", wtsHidtoOut, nHid * nOut);

    // 1.configure the network
    /* (note) Inhibitory neurons are used for the connection with negative weight value or
       Lateral inhibition of output layer  */
    // Create Input Layer 
    int gInExc = sim.createSpikeGeneratorGroup("inputexc", nIn, EXCITATORY_NEURON);
    int gInInh = sim.createSpikeGeneratorGroup("inputinh", nIn, INHIBITORY_NEURON);
    PoissonRate in(nIn);

    // Create Hidden and Output Layer
    int gHidExc = sim.createGroupMMBP("hiddenexc", nHid, EXCITATORY_NEURON);
    int gHidInh = sim.createGroupMMBP("hiddeninh", nHid, INHIBITORY_NEURON);
    int gOutInh = sim.createGroupMMBP("outputinh", nOut, INHIBITORY_NEURON);

    // setNeuromParametersMMBP(neuron group ID, tau_m, tau_ref, vTh, vReset, RangeRmem(rMem));
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
    
    sm_InExc = sim.setSpikeMonitor(gInExc, "DEFAULT");
    sm_InInh = sim.setSpikeMonitor(gInInh, "DEFAULT");
    sm_HidExc = sim.setSpikeMonitor(gHidExc, "DEFAULT");
    sm_HidInh = sim.setSpikeMonitor(gHidInh, "DEFAULT");
    sm_OutInh = sim.setSpikeMonitor(gOutInh, "DEFAULT");

    sim.setupNetwork();
    cm_InExc_HidExc = sim.setConnectionMonitor(gInExc, gHidExc, "DEFAULT");
    cm_InExc_HidInh = sim.setConnectionMonitor(gInExc, gHidInh, "DEFAULT");
    cm_InInh_HidExc = sim.setConnectionMonitor(gInInh, gHidExc, "DEFAULT");
    cm_InInh_HidInh = sim.setConnectionMonitor(gInInh, gHidInh, "DEFAULT");
    cm_HidInh_OutInh = sim.setConnectionMonitor(gHidInh, gOutInh, "DEFAULT");
    cm_HidExc_OutInh = sim.setConnectionMonitor(gHidExc, gOutInh, "DEFAULT");

    std::vector<int> rand_index;
    for(int i = 0; i < nTotalTestData; i++) { rand_index.push_back(i); }
    random_shuffle(rand_index.begin(), rand_index.end());

	int right = 0; 
	int wrong = 0;
    for(int test_index = 0; test_index < nTestData; test_index++) {
        //		int index = test_index;
        //		int index = int((float)rand()/RAND_MAX * nTestData);
        int rindex = rand_index[test_index];

        std::vector<int> v1 = intensity[rindex];
        std::vector<float> vinput(v1.begin(), v1.end());

        // ---------------- RUN STATE -------------------
        // 3. run network 
           in.setRates(0.0f);
           sim.setSpikeRate(gInExc,&in);
           sim.setSpikeRate(gInInh,&in);
           sim.runNetwork(rest/1000, rest%1000);

        sm_InExc->startRecording();
        sm_InInh->startRecording();
        sm_HidExc->startRecording();
        sm_HidInh->startRecording();
        sm_OutInh->startRecording();

        in.setRates(vinput);
        sim.setSpikeRate(gInExc,&in);
        sim.setSpikeRate(gInInh,&in);
        sim.runNetwork(run/1000, run%1000);

        sm_InExc->stopRecording();
        sm_InInh->stopRecording();
        sm_HidExc->stopRecording();
        sm_HidInh->stopRecording();
        sm_OutInh->stopRecording();

        int spikes[nOut];
        int maxS = -1, winner = -1;
        for(int k = 0; k < nOut; k++) {
            spikes[k] = sm_OutInh->getNeuronNumSpikes(k);
            if(maxS < spikes[k]) { maxS = spikes[k]; winner = k;}
        }
		if(winner == label[rindex]) right++;
		else wrong++;
        printf("%dth data, [PREDICTED | ANSWER] = [%d | %d], %d\n", rindex, winner, label[rindex], winner == label[rindex]);
    }

	printf("accuracy = %.4f\n", (float)(right)/(float)(right + wrong) * 100.0f);
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

static uint32_t swapU32(uint32_t value) {
    return(uint32_t)( (value & 0x000000ff) << 24 |
            (value & 0x0000ff00) <<  8 |
            (value & 0x00ff0000) >>  8 |
            (value & 0xff000000) >> 24 );
}

static void readDataHeader(std::ifstream& file, int &n_imgs, int &n_rows, int &n_cols) {
    int buffer = 0;

    file.read((char *) &buffer, sizeof(int));

    file.read((char *) &buffer, sizeof(int));
    n_imgs = (int) swapU32((uint32_t)buffer);

    file.read((char *) &buffer, sizeof(int));
    n_rows = (int) swapU32((uint32_t)buffer);

    file.read((char *) &buffer, sizeof(int));
    n_cols = (int) swapU32((uint32_t)buffer);
}

static void readLabelHeader(std::ifstream& file, int &n_imgs) {
    int buffer = 0;

    file.read((char *) &buffer, sizeof(int));

    file.read((char *) &buffer, sizeof(int));
    n_imgs = (int) swapU32((uint32_t)buffer);
}

static void readData(std::vector< std::vector<int> > &intensity) {
    std::ifstream file("mnist/t10k-images-idx3-ubyte");

    int n_imgs = 0; int n_rows = 0; int n_cols = 0;
    readDataHeader(file, n_imgs, n_rows, n_cols);

    for(int i = 0; i < n_imgs; ++i) {
        std::vector<int> v1;
        for(int r = 0; r < n_rows; ++r) {
            for(int c = 0; c < n_cols; ++c) {
                unsigned char temp = 0;
                file.read((char *) &temp, sizeof(temp));
                v1.push_back((int)temp);
            }
        }
        intensity.push_back(v1);
    }
    file.close();
}

static void readLabel(std::vector<int> &label) {
    std::ifstream file("mnist/t10k-labels-idx1-ubyte");

    int n_imgs = 0;
    readLabelHeader(file, n_imgs);

    for(int i = 0; i < n_imgs; ++i) {
        unsigned char temp = 0;
        file.read((char *) &temp, sizeof(temp));
        label.push_back(int(temp));
    }
    file.close();
}


