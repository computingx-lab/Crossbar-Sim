/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
* 
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Tile.h"
#include "Chip.h"
#include "ProcessingUnit.h"
#include "SubArray.h"
#include "Comparator.h"
#include "Adder.h"
#include "Definition.h"

using namespace std;

vector<vector<double> > getNetStructure(const string &inputfile);

int main(int argc, char * argv[]) {   

    auto start = chrono::high_resolution_clock::now();
    
    gen.seed(0);
    
    // ── CAM SubArray-only evaluation mode ─────────────────────────────────
    if (argc >= 2 && string(argv[1]) == "--cam-subarray") {
        if (argc < 8) {
            cout << "Usage: ./main --cam-subarray <weightfile> <inputfile> <numRows> <numCols> <numBitInput> <numBitSynapse> [adcBits]" << endl;
            return 1;
        }

        string weightfile  = argv[2];
        string inputfile   = argv[3];
        int    numRows     = atoi(argv[4]);
        int    numCols     = atoi(argv[5]);
        int    numBitInput = atoi(argv[6]);
        int    numBitSyn   = atoi(argv[7]);
        int    adcBits     = (argc >= 9) ? atoi(argv[8]) : 0;   // optional ADC precision (levelOutput = 2^adcBits)

        // Override params for this subarray BEFORE calling ProcessingUnitInitialize
        param->numRowSubArray   = numRows;
        param->numColSubArray   = numCols;
        param->numBitInput      = numBitInput;
        param->synapseBit       = numBitSyn;
        param->numRowPerSynapse = 1;
        param->numColPerSynapse = ceil((double)numBitSyn / (double)param->cellBit);
        // Parallel (analog) read: the whole column current is summed in one
        // shot and digitized by the multilevel ADC -- the physically correct
        // model for an analog crossbar MVM / inner-product, and the mode where
        // ADC precision (levelOutput = 2^adcBits) actually affects the cost.
        param->conventionalSequential = 0;
        param->conventionalParallel   = 1;
        param->parallelRead           = 1;
        param->numRowParallel         = numRows;   // all rows summed together
        param->numColMuxed            = 8;         // columns per ADC (NeuroSim parallel default)
        if (adcBits > 0) param->levelOutput = (int)pow(2, adcBits);  // ADC levels = 2^adc_bits

        // Let NeuroSim's own init function set up tech/cell/subArray correctly
        SubArray *subArray = nullptr;
        ProcessingUnitInitialize(subArray, inputParameter, tech, cell, 1, 1, 1, 1);

        // Load weight and input data (now safe — paths are correct, no argv[1] consumed)
        vector<vector<double>> newMemory = LoadInWeightData(
            weightfile, param->numRowPerSynapse, param->numColPerSynapse,
            param->maxConductance, param->minConductance);
        vector<vector<double>> inputVector = LoadInInputData(inputfile);

        double clkPeriod = 0;
        double totalReadLatency = 0;
        double totalReadEnergy  = 0;

        // Pass 1: determine clkPeriod (CalculateclkFreq = true)
        for (int k = 0; k < numBitInput; k++) {
            double activityRowRead = 0;
            vector<double> inputVec;
            for (int i = 0; i < (int)inputVector.size(); i++) {
                double x = (k < (int)inputVector[i].size()) ? inputVector[i][k] : 0.0;
                inputVec.push_back(x);
                if (x != 0) activityRowRead += 1.0;
            }
            activityRowRead /= numRows;
            subArray->activityRowRead = activityRowRead;

            vector<double> colRes = GetColumnResistance(inputVec, newMemory, cell,
                param->parallelRead, subArray->resCellAccess);
            subArray->CalculateLatency(1e20, colRes, true);
            if (clkPeriod < subArray->readLatency) clkPeriod = subArray->readLatency;
        }
        if (param->synchronous && clkPeriod > 0) param->clkFreq = 1.0 / clkPeriod;

        // Pass 2: actual latency + power (CalculateclkFreq = false)
        for (int k = 0; k < numBitInput; k++) {
            double activityRowRead = 0;
            vector<double> inputVec;
            for (int i = 0; i < (int)inputVector.size(); i++) {
                double x = (k < (int)inputVector[i].size()) ? inputVector[i][k] : 0.0;
                inputVec.push_back(x);
                if (x != 0) activityRowRead += 1.0;
            }
            activityRowRead /= numRows;
            subArray->activityRowRead = activityRowRead;

            vector<double> colRes = GetColumnResistance(inputVec, newMemory, cell,
                param->parallelRead, subArray->resCellAccess);
            subArray->CalculateLatency(1e20, colRes, false);
            subArray->CalculatePower(colRes);
            totalReadLatency += subArray->readLatency;
            totalReadEnergy  += subArray->readDynamicEnergy;
        }

        double subarrayArea  = subArray->area;
        double leakagePower  = subArray->leakage;

        cout << "\n─── CAM SubArray Hardware Evaluation ───────────────────" << endl;
        cout << "SubArray size        : " << numRows << " x " << numCols << endl;
        cout << "SubArray area        : " << subarrayArea * 1e12 << " um^2" << endl;
        cout << "Read latency (total) : " << totalReadLatency * 1e9 << " ns" << endl;
        cout << "Read energy  (total) : " << totalReadEnergy  * 1e12 << " pJ" << endl;
        cout << "Leakage power        : " << leakagePower * 1e6 << " uW" << endl;
        cout << "──────────────────────────────────────────────────────────" << endl;

        cout << "SYS_METRIC|AREA|"    << subarrayArea    << endl;
        cout << "SYS_METRIC|LATENCY|" << totalReadLatency << endl;
        cout << "SYS_METRIC|ENERGY|"  << totalReadEnergy  << endl;

        return 0;
    }
    // ── End CAM SubArray-only evaluation mode ──────────────────────────────

    // Comparator (top-k) cost evaluation mode
    // Reports the area / latency / energy of ONE comparison using NeuroSim's
    // own Comparator building block. PerfEval Part 2 (topk_cost.py) multiplies
    // this per-comparison cost by the number of comparisons a top-k selection
    // performs (~k*N), which is how the top-k cost grows with database size.
    //
    // Usage: ./main --comparator-cost <numBit>
    //   numBit = precision of the compared scores (e.g. ADC output bits).
    if (argc >= 2 && string(argv[1]) == "--comparator-cost") {
        int numBit = (argc >= 3) ? atoi(argv[2]) : 8;   // score precision
        if (numBit < 2) numBit = 2;   // Comparator latency model assumes >= 2 bits

        // Initialize tech/cell the same proven way the subarray mode does, so
        // the Technology the Comparator reads is fully set up. The comparator's
        // per-unit cost does not depend on array size, so defaults are fine.
        param->numRowSubArray        = 128;
        param->numColSubArray        = 128;
        param->numBitInput           = numBit;
        param->synapseBit            = 1;
        param->numRowPerSynapse      = 1;
        param->numColPerSynapse      = 1;
        param->conventionalSequential = 1;
        param->conventionalParallel   = 0;
        param->parallelRead           = 0;
        param->numRowParallel         = 1;
        param->numColMuxed            = 1;

        SubArray *subArray = nullptr;
        ProcessingUnitInitialize(subArray, inputParameter, tech, cell, 1, 1, 1, 1);

        // Build a single numBit-wide comparator on NeuroSim's Comparator block.
        // Call order matters: CalculateUnitArea() computes both the area AND the
        // internal gate capacitances that CalculateLatency()/CalculatePower()
        // then rely on, so it must run first (see Comparator.cpp).
        Comparator comparator(inputParameter, tech, cell);
        comparator.Initialize(numBit, 1);              // numBit-wide, one unit
        comparator.CalculateUnitArea(NONE);            // sets areaUnit + caps
        comparator.CalculateLatency(1e20, 0, 1);       // one comparison, unloaded
        comparator.CalculatePower(1, 1);               // one comparison

        double cmpArea    = comparator.areaUnit;             // m^2 per comparator
        double cmpLatency = comparator.readLatency;          // s per comparison
        double cmpEnergy  = comparator.readDynamicEnergy;    // J per comparison

        cout << "\nComparator (top-k) Cost Evaluation" << endl;
        cout << "Comparator bits       : " << numBit << endl;
        cout << "Area   (per unit)     : " << cmpArea    * 1e12 << " um^2" << endl;
        cout << "Latency(per compare)  : " << cmpLatency * 1e9  << " ns"  << endl;
        cout << "Energy (per compare)  : " << cmpEnergy  * 1e12 << " pJ"  << endl;

        cout << "CMP_METRIC|AREA|"    << cmpArea    << endl;
        cout << "CMP_METRIC|LATENCY|" << cmpLatency << endl;
        cout << "CMP_METRIC|ENERGY|"  << cmpEnergy  << endl;

        return 0;
    }
    // End Comparator cost evaluation mode


    // Adder (partial-sum merge) cost evaluation mode
    // Reports the area / latency / energy of ONE addition using NeuroSim's
    // own Adder block. PerfEval Part 1 uses this to cost recombining partial
    // dot products when the embedding dimension is split across arrays
    // (dim > array rows), i.e. the 'merge the pieces back together' step.
    //
    // Usage: ./main --adder-cost <numBit>
    //   numBit = width of the partial sums being added (e.g. ADC output bits).
    if (argc >= 2 && string(argv[1]) == "--adder-cost") {
        int numBit = (argc >= 3) ? atoi(argv[2]) : 8;   // width of partial sums
        if (numBit < 1) numBit = 1;

        // Initialize tech/cell the same proven way the other modes do.
        param->numRowSubArray        = 128;
        param->numColSubArray        = 128;
        param->numBitInput           = numBit;
        param->synapseBit            = 1;
        param->numRowPerSynapse      = 1;
        param->numColPerSynapse      = 1;
        param->conventionalSequential = 1;
        param->conventionalParallel   = 0;
        param->parallelRead           = 0;
        param->numRowParallel         = 1;
        param->numColMuxed            = 1;

        SubArray *subArray = nullptr;
        ProcessingUnitInitialize(subArray, inputParameter, tech, cell, 1, 1, 1, 1);

        // Build a single numBit-wide adder on NeuroSim's Adder block.
        // Call order matters: CalculateArea() computes both the area AND the
        // gate capacitances that CalculateLatency()/CalculatePower() rely on.
        Adder adder(inputParameter, tech, cell);
        adder.Initialize(numBit, 1, param->clkFreq);   // numBit-wide, one adder
        adder.CalculateArea(0, 0, NONE);               // sets area + caps
        adder.CalculateLatency(1e20, 0, 1);            // one addition, unloaded
        adder.CalculatePower(1, 1);                    // one addition

        double addArea    = adder.area;             // m^2 per adder
        double addLatency = adder.readLatency;      // s per addition
        double addEnergy  = adder.readDynamicEnergy;// J per addition

        cout << "\nAdder (partial-sum merge) Cost Evaluation" << endl;
        cout << "Adder bits            : " << numBit << endl;
        cout << "Area   (per unit)     : " << addArea    * 1e12 << " um^2" << endl;
        cout << "Latency(per add)      : " << addLatency * 1e9  << " ns"  << endl;
        cout << "Energy (per add)      : " << addEnergy  * 1e12 << " pJ"  << endl;

        cout << "ADD_METRIC|AREA|"    << addArea    << endl;
        cout << "ADD_METRIC|LATENCY|" << addLatency << endl;
        cout << "ADD_METRIC|ENERGY|"  << addEnergy  << endl;

        return 0;
    }
    // End Adder cost evaluation mode



    vector<vector<double> > netStructure;
    netStructure = getNetStructure(argv[1]);
    
    
    // define weight/input/memory precision from wrapper
    param->synapseBit = atoi(argv[2]);              // precision of synapse weight
    param->numBitInput = atoi(argv[3]);             // precision of input neural activation
    param->numRowSubArray = atoi(argv[4]);             // number row of subarray
    param->numRowParallel = atoi(argv[5]);             // number of enabled rows of subarray (partial parallel mode)


    if (param->cellBit > param->synapseBit) {
        cout << "ERROR!: Memory precision is even higher than synapse precision, please modify 'cellBit' in Param.cpp!" << endl;
        param->cellBit = param->synapseBit;
    }

    // 1.5 update: warning for the incompatible modes for a given device technology
    if ( param->memcelltype == 4 && param->technode !=22) {

        cout << "ERROR!: nvCap-CIM is only supported for 22 nm!" << endl;
        exit(-1);
    }
    

    if ( param->memcelltype == 2 && param->technode <=14) {

        cout << "ERROR!: RRAM-CIM is not supported beyond 22 nm!" << endl;
        exit(-1);
    }

    if ( param->deviceroadmap == 1 && param->technode <=14) {

        cout << "ERROR!: HP technology is not supported for 14 nm and beyond!" << endl;
        exit(-1);
    }

    if ( param->temp != 300 && param->technode <=14) {

        cout << "ERROR!: only 300K operation is supported for 14 nm and beyond!" << endl;
        exit(-1);
    }

    if ((param->sync_data_transfer)) {
    if (!((param->sync_data_transfer) && (param->globalBusType == true) && (param->chipActivation == true) 
    && (param->novelMapping == true) && (param->pipeline== true) 
    && (param->synchronous == true) && (param->globalBufferType == false)))

    {   cout << "ERROR!: sync_data_transfer mode not supported for the current parameter combimation!" << endl;
        exit(-1);
    }
    }

    if ((param->technode<=14) && (param->temp !=300))
    {
        cout << "WARNING!: TECHNOLOGY NODE UNDER 14NM IS ONLY SUPPORTED AT 300K TEMPERATURE!" << endl;
    }
    /*** initialize operationMode as default ***/
    param->conventionalParallel = 0;
    param->conventionalSequential = 0;
    param->BNNparallelMode = 0;                // parallel BNN
    param->BNNsequentialMode = 0;              // sequential BNN
    param->XNORsequentialMode = 0;           // Use several multi-bit RRAM as one synapse
    param->XNORparallelMode = 0;         // Use several multi-bit RRAM as one synapse
    switch(param->operationmode) {
        case 6:     param->XNORparallelMode = 1;               break;     
        case 5:     param->XNORsequentialMode = 1;             break;     
        case 4:     param->BNNparallelMode = 1;                break;     
        case 3:     param->BNNsequentialMode = 1;              break;    
        case 2:     param->conventionalParallel = 1;           break;     
        case 1:     param->conventionalSequential = 1;         break;    
        case -1:    break;
        default:    exit(-1);
    }
    
    if (param->XNORparallelMode || param->XNORsequentialMode) {
        param->numRowPerSynapse = 2;
    } else {
        param->numRowPerSynapse = 1;
    }
    if (param->BNNparallelMode) {
        param->numColPerSynapse = 2;
    } else if (param->XNORparallelMode || param->XNORsequentialMode || param->BNNsequentialMode) {
        param->numColPerSynapse = 1;
    } else {
        param->numColPerSynapse = ceil((double)param->synapseBit/(double)param->cellBit); 
    }
    

    // 1.4 update : Implementation for conventional sequential 
    if ( ( param->conventionalSequential == 1 || param->conventionalSequential == 3 || param->conventionalSequential == 5 ) && (param->memcelltype==1) )
    {
    param->numColMuxed = param->numColPerSynapse;
    }

    double maxPESizeNM, maxTileSizeCM, numPENM;

    maxPESizeNM = 0;
    maxTileSizeCM = 0;
    numPENM = 0;

    vector<int> markNM;
    vector<int> pipelineSpeedUp;
    markNM = ChipDesignInitialize(inputParameter, tech, cell, false, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
    pipelineSpeedUp = ChipDesignInitialize(inputParameter, tech, cell, false, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
    
    if (maxPESizeNM == 0 || maxTileSizeCM == 0) {
    cout << "\n[WARNING] ChipDesignInitialize skipped constraint optimization (common for single-layer workloads)." << endl;

    // Floor to minimum viable hierarchy values based on subarray size
    if (maxPESizeNM < 2 * param->numRowSubArray) {
        maxPESizeNM = 2 * param->numRowSubArray;   // minimum 2 subarrays per PE
    }
    if (maxTileSizeCM < 4 * param->numRowSubArray) {
        maxTileSizeCM = 4 * param->numRowSubArray; // minimum 4 subarrays per tile
    }

    cout << "[INFO] Applied default hardware structural constraints: maxPESizeNM = " 
         << maxPESizeNM << ", maxTileSizeCM = " << maxTileSizeCM << "\n" << endl;
    }

    double desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM;
    int numTileRow, numTileCol;
    
    vector<vector<double> > numTileEachLayer;
    vector<vector<double> > utilizationEachLayer;
    vector<vector<double> > speedUpEachLayer;
    vector<vector<double> > tileLocaEachLayer;
    
    numTileEachLayer = ChipFloorPlan(true, false, false, netStructure, markNM, 
                    maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
                    &desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);    
    
    utilizationEachLayer = ChipFloorPlan(false, true, false, netStructure, markNM, 
                    maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
                    &desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);
    
    speedUpEachLayer = ChipFloorPlan(false, false, true, netStructure, markNM,
                    maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
                    &desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);
                    
    tileLocaEachLayer = ChipFloorPlan(false, false, false, netStructure, markNM,
                    maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
                    &desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);
    
    cout << "------------------------------ FloorPlan --------------------------------" <<  endl;
    cout << endl;
    cout << "Tile and PE size are optimized to maximize memory utilization ( = memory mapped by synapse / total memory on chip)" << endl;
    cout << endl;
    if (!param->novelMapping) {
        cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
        cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
    } else {
        cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
        cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
        cout << "Desired Novel Mapped Tile Storage Size: " << numPENM << "x" << desiredPESizeNM << "x" << desiredPESizeNM << endl;
    }
    cout << "User-defined SubArray Size: " << param->numRowSubArray << "x" << param->numColSubArray << endl;
    cout << endl;
    cout << "----------------- # of tile used for each layer -----------------" <<  endl;
    double totalNumTile = 0;
    for (int i=0; i<netStructure.size(); i++) {
        cout << "layer" << i+1 << ": " << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
        totalNumTile += numTileEachLayer[0][i] * numTileEachLayer[1][i];
    }
    cout << endl;

    cout << "----------------- Speed-up of each layer ------------------" <<  endl;
    for (int i=0; i<netStructure.size(); i++) {
        cout << "layer" << i+1 << ": " << speedUpEachLayer[0][i] * speedUpEachLayer[1][i] << endl;
    }
    cout << endl;
    
    cout << "----------------- Utilization of each layer ------------------" <<  endl;
    double realMappedMemory = 0;
    for (int i=0; i<netStructure.size(); i++) {
        cout << "layer" << i+1 << ": " << utilizationEachLayer[i][0] << endl;
        realMappedMemory += numTileEachLayer[0][i] * numTileEachLayer[1][i] * utilizationEachLayer[i][0];
    }
    cout << "Memory Utilization of Whole Chip: " << realMappedMemory/totalNumTile*100 << " % " << endl;
    cout << endl;
    cout << "---------------------------- FloorPlan Done ------------------------------" <<  endl;
    cout << endl;
    cout << endl;
    cout << endl;
    
    double numComputation = 0;
    for (int i=0; i<netStructure.size(); i++) {
        numComputation += 2*(netStructure[i][0] * netStructure[i][1] * netStructure[i][2] * netStructure[i][3] * netStructure[i][4] * netStructure[i][5]);
    }

    
    ChipInitialize(inputParameter, tech, cell, netStructure, markNM, numTileEachLayer,
                    numPENM, desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow, numTileCol);
            
    double chipHeight, chipWidth, chipArea, chipAreaIC, chipAreaADC, chipAreaAccum, chipAreaOther, chipAreaArray;
    double CMTileheight = 0;
    double CMTilewidth = 0;
    double NMTileheight = 0;
    double NMTilewidth = 0;
    vector<double> chipAreaResults;
    
    chipAreaResults = ChipCalculateArea(inputParameter, tech, cell, desiredNumTileNM, numPENM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow, 
                    &chipHeight, &chipWidth, &CMTileheight, &CMTilewidth, &NMTileheight, &NMTilewidth);     
    
    chipArea = chipAreaResults[0];
    chipAreaIC = chipAreaResults[1];
    chipAreaADC = chipAreaResults[2];
    chipAreaAccum = chipAreaResults[3];
    chipAreaOther = chipAreaResults[4];
    chipAreaArray = chipAreaResults[5];

    double clkPeriod = 0;
    double layerclkPeriod = 0;
    
    double chipReadLatency = 0;
    double chipReadDynamicEnergy = 0;
    double chipLeakageEnergy = 0;
    double chipLeakage = 0;
    double chipbufferLatency = 0;
    double chipbufferReadDynamicEnergy = 0;
    double chipicLatency = 0;
    double chipicReadDynamicEnergy = 0;
    
    double chipLatencyADC = 0;
    double chipLatencyAccum = 0;
    double chipLatencyOther = 0;
    double chipEnergyADC = 0;
    double chipEnergyAccum = 0;
    double chipEnergyOther = 0;
    
    double layerReadLatency = 0;
    double layerReadDynamicEnergy = 0;
    double tileLeakage = 0;
    // Anni update: leakage of SRAM when partial subarray in use
    double tileLeakageSRAMInUse = 0;
    double layerbufferLatency = 0;
    double layerbufferDynamicEnergy = 0;
    double layericLatency = 0;
    double layericDynamicEnergy = 0;
    
    double coreLatencyADC = 0;
    double coreLatencyAccum = 0;
    double coreLatencyOther = 0;
    double coreEnergyADC = 0;
    double coreEnergyAccum = 0;
    double coreEnergyOther = 0;
    
    if (param->synchronous){
        // calculate clkFreq
        for (int i=0; i<netStructure.size(); i++) {     
            // Anni update: add &tileLeakageSRAMInUse
            ChipCalculatePerformance(inputParameter, tech, cell, i, argv[2*i+6], argv[2*i+7], argv[2*i+8], netStructure[i][6],
                        netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
                        numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
                        &layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &tileLeakageSRAMInUse, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
                        &coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, true, &layerclkPeriod);
            if(clkPeriod < layerclkPeriod){
                clkPeriod = layerclkPeriod;
            }           
        }       
        cout<<"clkPeriod: "<<clkPeriod<<endl;
        if(param->clkFreq > 1/clkPeriod){
            param->clkFreq = 1/clkPeriod;
        }
    }

    cout << "-------------------------------------- Hardware Performance --------------------------------------" <<  endl;  
    if (! param->pipeline) {
        // layer-by-layer process
        // show the detailed hardware performance for each layer
        for (int i=0; i<netStructure.size(); i++) {
            cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;
            // Anni update: add &tileLeakageSRAMInUse
            ChipCalculatePerformance(inputParameter, tech, cell, i, argv[2*i+6], argv[2*i+7], argv[2*i+8], netStructure[i][6],
                        netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
                        numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
                        &layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &tileLeakageSRAMInUse, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
                        &coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, false, &layerclkPeriod);
            if (param->synchronous) {
                layerReadLatency *= clkPeriod;
                layerbufferLatency *= clkPeriod;
                layericLatency *= clkPeriod;
                coreLatencyADC *= clkPeriod;
                coreLatencyAccum *= clkPeriod;
                coreLatencyOther *= clkPeriod;
            }
            
            double numTileOtherLayer = 0;
            double layerLeakageEnergy = 0;      
            for (int j=0; j<netStructure.size(); j++) {
                if (j != i) {
                    numTileOtherLayer += numTileEachLayer[0][j] * numTileEachLayer[1][j];
                }
            }
            // Anni update: other layer tiles and partial this layer tiles are in leakage
            layerLeakageEnergy = (numTileOtherLayer * tileLeakage + numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakageSRAMInUse) * layerReadLatency;
            
            cout << "layer" << i+1 << "'s readLatency is: " << layerReadLatency*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s readDynamicEnergy is: " << layerReadDynamicEnergy*1e12 << "pJ" << endl;
            cout << "layer" << i+1 << "'s leakagePower is: " << numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage*1e6 << "uW" << endl;
            cout << "layer" << i+1 << "'s leakageEnergy is: " << layerLeakageEnergy*1e12 << "pJ" << endl;
            cout << "layer" << i+1 << "'s buffer latency is: " << layerbufferLatency*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << layerbufferDynamicEnergy*1e12 << "pJ" << endl;
            cout << "layer" << i+1 << "'s ic latency is: " << layericLatency*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s ic readDynamicEnergy is: " << layericDynamicEnergy*1e12 << "pJ" << endl;
            
            
            cout << endl;
            cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
            cout << endl;
            cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADC*1e9 << "ns" << endl;
            cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccum*1e9 << "ns" << endl;
            cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOther*1e9 << "ns" << endl;
            cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADC*1e12 << "pJ" << endl;
            cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccum*1e12 << "pJ" << endl;
            cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOther*1e12 << "pJ" << endl;
            cout << endl;
            cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
            cout << endl;
            
            chipReadLatency += layerReadLatency;
            chipReadDynamicEnergy += layerReadDynamicEnergy;
            chipLeakageEnergy += layerLeakageEnergy;
            chipLeakage += tileLeakage*numTileEachLayer[0][i] * numTileEachLayer[1][i];
            chipbufferLatency += layerbufferLatency;
            chipbufferReadDynamicEnergy += layerbufferDynamicEnergy;
            chipicLatency += layericLatency;
            chipicReadDynamicEnergy += layericDynamicEnergy;
            
            chipLatencyADC += coreLatencyADC;
            chipLatencyAccum += coreLatencyAccum;
            chipLatencyOther += coreLatencyOther;
            chipEnergyADC += coreEnergyADC;
            chipEnergyAccum += coreEnergyAccum;
            chipEnergyOther += coreEnergyOther;
        }
    } else {
        // pipeline system
        // firstly define system clock
        double systemClock = 0;
        
        vector<double> readLatencyPerLayer;
        vector<double> readDynamicEnergyPerLayer;
        vector<double> leakagePowerPerLayer;
        vector<double> bufferLatencyPerLayer;
        vector<double> bufferEnergyPerLayer;
        vector<double> icLatencyPerLayer;
        vector<double> icEnergyPerLayer;
        
        vector<double> coreLatencyADCPerLayer;
        vector<double> coreEnergyADCPerLayer;
        vector<double> coreLatencyAccumPerLayer;
        vector<double> coreEnergyAccumPerLayer;
        vector<double> coreLatencyOtherPerLayer;
        vector<double> coreEnergyOtherPerLayer;
        
        for (int i=0; i<netStructure.size(); i++) {
            // Anni update: add &tileLeakageSRAMInUse
            ChipCalculatePerformance(inputParameter, tech, cell, i, argv[2*i+6], argv[2*i+7], argv[2*i+8], netStructure[i][6],
                        netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
                        numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
                        &layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &tileLeakageSRAMInUse, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
                        &coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, false, &layerclkPeriod);
            if (param->synchronous) {
                layerReadLatency *= clkPeriod;
                layerbufferLatency *= clkPeriod;
                layericLatency *= clkPeriod;
                coreLatencyADC *= clkPeriod;
                coreLatencyAccum *= clkPeriod;
                coreLatencyOther *= clkPeriod;
            }           
            
            systemClock = MAX(systemClock, layerReadLatency);
            
            readLatencyPerLayer.push_back(layerReadLatency);
            readDynamicEnergyPerLayer.push_back(layerReadDynamicEnergy);
            // Anni update: average leakage power considering read latency
            leakagePowerPerLayer.push_back(numTileEachLayer[0][i] * numTileEachLayer[1][i] * (tileLeakage * (systemClock-readLatencyPerLayer[i]) + tileLeakageSRAMInUse * readLatencyPerLayer[i]) / systemClock);
            bufferLatencyPerLayer.push_back(layerbufferLatency);
            bufferEnergyPerLayer.push_back(layerbufferDynamicEnergy);
            icLatencyPerLayer.push_back(layericLatency);
            icEnergyPerLayer.push_back(layericDynamicEnergy);
            
            coreLatencyADCPerLayer.push_back(coreLatencyADC);
            coreEnergyADCPerLayer.push_back(coreEnergyADC);
            coreLatencyAccumPerLayer.push_back(coreLatencyAccum);
            coreEnergyAccumPerLayer.push_back(coreEnergyAccum);
            coreLatencyOtherPerLayer.push_back(coreLatencyOther);
            coreEnergyOtherPerLayer.push_back(coreEnergyOther);
        }
        
        for (int i=0; i<netStructure.size(); i++) {
            
            cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;

            cout << "layer" << i+1 << "'s readLatency is: " << readLatencyPerLayer[i]*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s readDynamicEnergy is: " << readDynamicEnergyPerLayer[i]*1e12 << "pJ" << endl;
            cout << "layer" << i+1 << "'s leakagePower is: " << leakagePowerPerLayer[i]*1e6 << "uW" << endl;
            // Anni update: average leakage power considering read latency
            cout << "layer" << i+1 << "'s leakageEnergy is: " << leakagePowerPerLayer[i] * systemClock *1e12 << "pJ" << endl;
            cout << "layer" << i+1 << "'s buffer latency is: " << bufferLatencyPerLayer[i]*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << bufferEnergyPerLayer[i]*1e12 << "pJ" << endl;
            cout << "layer" << i+1 << "'s ic latency is: " << icLatencyPerLayer[i]*1e9 << "ns" << endl;
            cout << "layer" << i+1 << "'s ic readDynamicEnergy is: " << icEnergyPerLayer[i]*1e12 << "pJ" << endl;

            cout << endl;
            cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
            cout << endl;
            cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADCPerLayer[i]*1e9 << "ns" << endl;
            cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccumPerLayer[i]*1e9 << "ns" << endl;
            cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOtherPerLayer[i]*1e9 << "ns" << endl;
            cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADCPerLayer[i]*1e12 << "pJ" << endl;
            cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccumPerLayer[i]*1e12 << "pJ" << endl;
            cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOtherPerLayer[i]*1e12 << "pJ" << endl;
            cout << endl;
            cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
            cout << endl;
            
            chipReadLatency = systemClock;
            chipReadDynamicEnergy += readDynamicEnergyPerLayer[i];
            // Anni update: average leakage power considering read latency
            chipLeakageEnergy += leakagePowerPerLayer[i] * systemClock;
            chipLeakage += leakagePowerPerLayer[i];
            chipbufferLatency = MAX(chipbufferLatency, bufferLatencyPerLayer[i]);
            chipbufferReadDynamicEnergy += bufferEnergyPerLayer[i];
            chipicLatency = MAX(chipicLatency, icLatencyPerLayer[i]);
            chipicReadDynamicEnergy += icEnergyPerLayer[i];
            
            chipLatencyADC = MAX(chipLatencyADC, coreLatencyADCPerLayer[i]);
            chipLatencyAccum = MAX(chipLatencyAccum, coreLatencyAccumPerLayer[i]);
            chipLatencyOther = MAX(chipLatencyOther, coreLatencyOtherPerLayer[i]);
            chipEnergyADC += coreEnergyADCPerLayer[i];
            chipEnergyAccum += coreEnergyAccumPerLayer[i];
            chipEnergyOther += coreEnergyOtherPerLayer[i];
        }
        
    }
    
    cout << "------------------------------ Summary --------------------------------" <<  endl;
    cout << endl;
    cout << "ChipArea : " << chipArea*1e12 << "um^2" << endl;
    cout << "Chip total CIM array : " << chipAreaArray*1e12 << "um^2" << endl;
    cout << "Total IC Area on chip (Global and Tile/PE local): " << chipAreaIC*1e12 << "um^2" << endl;
    cout << "Total ADC (or S/As and precharger for SRAM) Area on chip : " << chipAreaADC*1e12 << "um^2" << endl;
    cout << "Total Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) on chip : " << chipAreaAccum*1e12 << "um^2" << endl;
    cout << "Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, pooling and activation units) : " << chipAreaOther*1e12 << "um^2" << endl;
    cout << endl;
    if (! param->pipeline) {
        if (param->synchronous) cout << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
        cout << "Chip layer-by-layer readLatency (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
        cout << "Chip total readDynamicEnergy is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
        cout << "Chip total leakage Energy is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
        cout << "Chip total leakage Power is: " << chipLeakage*1e6 << "uW" << endl;
        cout << "Chip buffer readLatency is: " << chipbufferLatency*1e9 << "ns" << endl;
        cout << "Chip buffer readDynamicEnergy is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
        cout << "Chip ic readLatency is: " << chipicLatency*1e9 << "ns" << endl;
        cout << "Chip ic readDynamicEnergy is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
    } else {
        if (param->synchronous) cout << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
        cout << "Chip pipeline-system-clock-cycle (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
        cout << "Chip pipeline-system readDynamicEnergy (per image) is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
        cout << "Chip pipeline-system leakage Energy (per image) is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
        cout << "Chip pipeline-system leakage Power (per image) is: " << chipLeakage*1e6 << "uW" << endl;
        cout << "Chip pipeline-system buffer readLatency (per image) is: " << chipbufferLatency*1e9 << "ns" << endl;
        cout << "Chip pipeline-system buffer readDynamicEnergy (per image) is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
        cout << "Chip pipeline-system ic readLatency (per image) is: " << chipicLatency*1e9 << "ns" << endl;
        cout << "Chip pipeline-system ic readDynamicEnergy (per image) is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
    }
    
    cout << endl;
    cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
    cout << endl;
    cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << chipLatencyADC*1e9 << "ns" << endl;
    cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << chipLatencyAccum*1e9 << "ns" << endl;
    cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << chipLatencyOther*1e9 << "ns" << endl;
    cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << chipEnergyADC*1e12 << "pJ" << endl;
    cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << chipEnergyAccum*1e12 << "pJ" << endl;
    cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << chipEnergyOther*1e12 << "pJ" << endl;
    cout << endl;
    cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
    cout << endl;
    
    cout << endl;
    cout << "----------------------------- Performance -------------------------------" << endl;
    if (! param->pipeline) {
        if(param->validated){
            cout << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl;    // post-layout energy increase, zeta = 1.23 by default
        }else{
            cout << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
        }
        cout << "Throughput TOPS (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12) << endl;
        cout << "Throughput FPS (Layer-by-Layer Process): " << 1/(chipReadLatency) << endl;
        cout << "Compute efficiency TOPS/mm^2 (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
    } else {
        if(param->validated){
            cout << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl; // post-layout energy increase, zeta = 1.23 by default
        }else{
            cout << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
        }
        cout << "Throughput TOPS (Pipelined Process): " << numComputation/(chipReadLatency*1e12) << endl;
        cout << "Throughput FPS (Pipelined Process): " << 1/(chipReadLatency) << endl;
        cout << "Compute efficiency TOPS/mm^2 (Pipelined Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
    }
    cout << "-------------------------------------- Hardware Performance Done --------------------------------------" <<  endl;
    cout << endl;
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(stop-start);
    cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
    cout << "Total Run-time of NeuroSim: " << duration.count() << " seconds" << endl;
    cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
    
    // debugging code

    /*
    fstream read;
    // read.open("filelocation/filename",fstream::app);    
    read.open("/home/junmo/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/Data_TechnologyUpdate/overall_metrics.csv",fstream::app); 
    
    // enter the filelocation/filename where you want to store the printed values. 

    read<<param->technode<<", " ;
    read<<param->operationmode<<", ";
    read<<param->memcelltype<<", ";
    read<<param->accesstype<<", ";

    read<<chipArea*1e12<<", "<<chipAreaArray*1e12<<", "<<chipAreaIC*1e12<<", ";
    read<<chipAreaADC*1e12<<", "<<chipAreaAccum*1e12<<", "<<chipAreaOther*1e12<<", ";
    read<<clkPeriod*1e9<<", ";
    read<<chipReadLatency*1e9<<", ";
    read<<chipReadDynamicEnergy*1e12<<", ";
    read<<chipLeakageEnergy*1e12<<", ";
    read<<chipLeakage*1e6<<", ";

    read<<chipbufferLatency*1e9<<", ";
    read<<chipbufferReadDynamicEnergy*1e12<<", ";
    read<<chipicLatency*1e9 <<", ";
    read<<chipicReadDynamicEnergy*1e12 <<", ";

    read<<chipLatencyADC*1e9<<", ";
    read<<chipLatencyAccum*1e9 <<", ";
    read<<chipLatencyOther*1e9 <<", ";
    read<<chipEnergyADC*1e12 <<", ";
    read<<chipEnergyAccum*1e12<<", ";
    read<<chipEnergyOther*1e12<<", ";

    read<<numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta<<", ";
    read<<numComputation/(chipReadLatency*1e12)<<", ";
    read<<1/(chipReadLatency) <<", ";
    read<<numComputation/(chipReadLatency*1e12)/(chipArea*1e6) <<", ";
    read<<numComputation<<", ";
    read<< "param->inputtoggle"<<", "<<param->inputtoggle<<", ";
    read<< "param->numRowParallel"<<", "<<param->numRowParallel<<", ";
    read<< "onoff"<<", "<<param->resistanceOn/param->resistanceOff<<", ";
    read<< "levelOutput"<<", "<<param->levelOutput <<", ";
    read<< "CellBit"<<", "<<param->cellBit <<", ";
    read<< "ADCcurrentmode" <<", "<<param->currentMode<<", ";
    read<< "SubArraySize" <<", "<<param->numRowSubArray<<", ";
    read<< "ADCdelay" <<", "<<param->ADClatency<<", ";
    read<< "rowdelay" <<", "<<param->rowdelay<<", ";
    read<< "muxdelay" <<", "<<param->muxdelay<<", ";
    read<<endl;

    */

    // --- BEGIN AUTOMATION TELEMETRY HOOK ---
    std::cout << "SYS_METRIC|AREA|" << chipArea << std::endl;
    std::cout << "SYS_METRIC|LATENCY|" << chipReadLatency << std::endl;
    std::cout << "SYS_METRIC|ENERGY|" << chipReadDynamicEnergy << std::endl;
    // --- END AUTOMATION TELEMETRY HOOK ---
    
    return 0;
}

vector<vector<double> > getNetStructure(const string &inputfile) {
    ifstream infile(inputfile.c_str());      
    string inputline;
    string inputval;
    
    int ROWin=0, COLin=0;      
    if (!infile.good()) {        
        cerr << "Error: the input file cannot be opened!" << endl;
        exit(1);
    }else{
        while (getline(infile, inputline, '\n')) {       
            ROWin++;                                
        }
        infile.clear();
        infile.seekg(0, ios::beg);      
        if (getline(infile, inputline, '\n')) {        
            istringstream iss (inputline);      
            while (getline(iss, inputval, ',')) {       
                COLin++;
            }
        }   
    }
    infile.clear();
    infile.seekg(0, ios::beg);          

    vector<vector<double> > netStructure;               
    for (int row=0; row<ROWin; row++) { 
        vector<double> netStructurerow;
        getline(infile, inputline, '\n');             
        istringstream iss;
        iss.str(inputline);
        for (int col=0; col<COLin; col++) {       
            while(getline(iss, inputval, ',')){ 
                istringstream fs;
                fs.str(inputval);
                double f=0;
                fs >> f;                
                netStructurerow.push_back(f);           
            }           
        }       
        netStructure.push_back(netStructurerow);
    }
    infile.close();
    
    return netStructure;
    netStructure.clear();
}   