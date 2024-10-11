#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <cstring>

typedef hls::vector<float,16> float16;
typedef hls::vector<float,8> float8;
typedef hls::vector<float,4> float4;
typedef hls::vector<float,2> float2;
typedef hls::vector<float,1> float1;

void load_input (float input[1][228][228], float4 vinput[3326976], int d0) {
#pragma HLS inline off
    for (int i0 = 0; i0 < 1; i0+=1){
        for (int i1 = 0; i1 < 228; i1+=1){
            for (int i2 = 0; i2 < 228; i2+=4){
#pragma HLS pipeline II=1
                float4 tmp_input = vinput[(i0 + d0 * 1) * 12996 + i1 * 57 + i2/4];
                input[i0][i1][i2 + 0] = tmp_input[0];
                input[i0][i1][i2 + 1] = tmp_input[1];
                input[i0][i1][i2 + 2] = tmp_input[2];
                input[i0][i1][i2 + 3] = tmp_input[3];
            }
        }
    }
}
void load_output (float output[16][224][224], float16 voutput[802816], int d0) {
#pragma HLS inline off
    for (int i0 = 0; i0 < 16; i0+=1){
        for (int i1 = 0; i1 < 224; i1+=1){
            for (int i2 = 0; i2 < 224; i2+=16){
#pragma HLS pipeline II=1
                float16 tmp_output = voutput[(i0 + d0 * 16) * 3136 + i1 * 14 + i2/16];
                output[i0][i1][i2 + 0] = tmp_output[0];
                output[i0][i1][i2 + 1] = tmp_output[1];
                output[i0][i1][i2 + 2] = tmp_output[2];
                output[i0][i1][i2 + 3] = tmp_output[3];
                output[i0][i1][i2 + 4] = tmp_output[4];
                output[i0][i1][i2 + 5] = tmp_output[5];
                output[i0][i1][i2 + 6] = tmp_output[6];
                output[i0][i1][i2 + 7] = tmp_output[7];
                output[i0][i1][i2 + 8] = tmp_output[8];
                output[i0][i1][i2 + 9] = tmp_output[9];
                output[i0][i1][i2 + 10] = tmp_output[10];
                output[i0][i1][i2 + 11] = tmp_output[11];
                output[i0][i1][i2 + 12] = tmp_output[12];
                output[i0][i1][i2 + 13] = tmp_output[13];
                output[i0][i1][i2 + 14] = tmp_output[14];
                output[i0][i1][i2 + 15] = tmp_output[15];
            }
        }
    }
}
void load_weight (float weight[16][256][5][5], float1 vweight[1638400], int d0) {
#pragma HLS inline off
    for (int i0 = 0; i0 < 16; i0+=1){
        for (int i1 = 0; i1 < 256; i1+=1){
            for (int i2 = 0; i2 < 5; i2+=1){
                for (int i3 = 0; i3 < 5; i3+=1){
#pragma HLS pipeline II=1
                    float1 tmp_weight = vweight[(i0 + d0 * 16) * 6400 + i1 * 25 + i2 * 5 + i3/1];
                    weight[i0][i1][i2][i3 + 0] = tmp_weight[0];
                }
            }
        }
    }
}
void store_output (float output[16][224][224], float16 voutput[802816], int d0) {
#pragma HLS inline off
    for (int i0 = 0; i0 < 16; i0+=1){
        for (int i1 = 0; i1 < 224; i1+=1){
            for (int i2 = 0; i2 < 224; i2+=16){
#pragma HLS pipeline II=1
                float16 tmp_output;
                tmp_output[0] = output[i0][i1][i2 + 0];
                tmp_output[1] = output[i0][i1][i2 + 1];
                tmp_output[2] = output[i0][i1][i2 + 2];
                tmp_output[3] = output[i0][i1][i2 + 3];
                tmp_output[4] = output[i0][i1][i2 + 4];
                tmp_output[5] = output[i0][i1][i2 + 5];
                tmp_output[6] = output[i0][i1][i2 + 6];
                tmp_output[7] = output[i0][i1][i2 + 7];
                tmp_output[8] = output[i0][i1][i2 + 8];
                tmp_output[9] = output[i0][i1][i2 + 9];
                tmp_output[10] = output[i0][i1][i2 + 10];
                tmp_output[11] = output[i0][i1][i2 + 11];
                tmp_output[12] = output[i0][i1][i2 + 12];
                tmp_output[13] = output[i0][i1][i2 + 13];
                tmp_output[14] = output[i0][i1][i2 + 14];
                tmp_output[15] = output[i0][i1][i2 + 15];
                voutput[(i0 + d0 * 16) * 3136 + i1 * 14 + i2/16] = tmp_output;
            }
        }
    }
}

void cnn_layer(float input[1][228][228], float output[16][224][224], float weight[16][256][5][5], float4 vinput[3326976], float1 vweight[1638400], float16 voutput[802816]) {
    int i;
    for (int i0 = 0; i0 < 16; i0++) {
        load_weight(weight, vweight, i0);
        load_output(output, voutput, i0);
        for (int j = 0; j < 256; j++) {
            load_input(input, vinput, j);
            for (int h = 0; h < 224; h++) {
                for (int w = 0; w < 224; w++) {
#pragma HLS pipeline II=1
                    for (int i2 = 0; i2 < 16; i2++) {
                        for (int q = 0; q < 5; q++) {
                            for (int p = 0; p < 5; p++) {
                                i = i0 * 16 + i2;
                                output[i2][h][w] +=weight[i2][j][p][q] * input[0][h+p][w+q] ;
                            }
                        }
                    }
                }
            }
        }
        store_output(output, voutput, i0);
    }
}

void kernel_cnn(float4 vinput[3326976], float1 vweight[1638400], float16 voutput[802816]) {
    
#pragma HLS INTERFACE m_axi port=vinput offset=slave bundle=kernel_input
#pragma HLS INTERFACE m_axi port=voutput offset=slave bundle=kernel_output
#pragma HLS INTERFACE m_axi port=vweight offset=slave bundle=kernel_weight
#pragma HLS INTERFACE s_axilite port=vinput bundle=control
#pragma HLS INTERFACE s_axilite port=voutput bundle=control
#pragma HLS INTERFACE s_axilite port=vweight bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    
    float input[1][228][228];
    float output[16][224][224];
    float weight[16][256][5][5];
    
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=1 dim=1
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=6 dim=2
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=6 dim=3
    
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=1 dim=2
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=16 dim=3
    
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=1 dim=2
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=5 dim=3
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=5 dim=4
    
    cnn_layer(input, output, weight, vinput, vweight, voutput);
}
