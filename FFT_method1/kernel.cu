#include <iostream>
#include <cstdio>
#include <complex>
#include <cmath>
#include <iterator>
#include "string.h"
#include <chrono>
#include <memory>
#include "parameters.h"
#include <thrust/complex.h>

using namespace std;

//call external GPU_Kernel from .cu
extern void GPU_kernel(int* v, IndexSave* indsave, thrust::complex<double>* fft_);

//const naming
const double PI = 3.1415926536;
const int samp = SIZE;
const int f_size = 256;

//reserve bits for fft processing
unsigned int bitReverse(unsigned int x, int log2n) {
    int n = 0;
    for (int i = 0; i < log2n; i++) {
        if (x & (1 << i)) n |= 1 << (log2n - 1 - i);
    }
    return n;
}

//hamming_window
void Hamming_Window(double window[], int frame_size) {
    for (int i = 0; i < frame_size; i++) window[i] = 0.54f - 0.46f * cos((float)i * 2.0f * PI / float((frame_size - 1)));
}

//fft
template<class Iter_T>
void fft(Iter_T a, Iter_T b, int log2n)
{
    typedef typename iterator_traits<Iter_T>::value_type complex;
    const complex J(0, 1);
    int n = 1 << log2n;

    for (int i = 0; i < n; ++i) b[bitReverse(i, log2n)] = a[i];
    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        complex w(1, 0);
        complex wm = exp(-J * (PI / m2));
        for (int j = 0; j < m2; ++j) {
            for (int k = j; k < n; k += m) {
                complex t = w * b[k + m2];
                complex u = b[k];
                b[k] = u + t;
                b[k + m2] = u - t;
                w *= wm;
            }
        }
    }
}

//Wav Header
struct wav_header_t
{
    char chunkID[4]; //"RIFF" = 0x46464952
    unsigned long chunkSize; //28 [+ sizeof(wExtraFormatBytes) + wExtraFormatBytes] + sum(sizeof(chunk.id) + sizeof(chunk.size) + chunk.size)
    char format[4]; //"WAVE" = 0x45564157
    char subchunk1ID[4]; //"fmt " = 0x20746D66
    unsigned long subchunk1Size; //16 [+ sizeof(wExtraFormatBytes) + wExtraFormatBytes]
    unsigned short audioFormat;
    unsigned short numChannels;
    unsigned long sampleRate;
    unsigned long byteRate;
    unsigned short blockAlign;
    unsigned short bitsPerSample;
};

//Chunks
struct chunk_t
{
    char ID[4]; //"data" = 0x61746164
    unsigned long size;  //Chunk data bytes
};

int* process()
{
    const char* fileName = "re.wav";
    const char* fileToSave = "result_method1.dat";
    FILE* fin = fopen(fileName, "rb");

    //Read WAV header
    wav_header_t header;

    fread(&header, sizeof(header), 1, fin);
    printf(".......................................................\n");
    printf("Below is the information of .wav\n");

    //Print WAV header
    printf("WAV File Header read:\n");
    printf("File Type: %s\n", header.chunkID);
    printf("File Size: %ld\n", header.chunkSize);
    printf("WAV Marker: %s\n", header.format);
    printf("Format Name: %s\n", header.subchunk1ID);
    printf("Format Length: %ld\n", header.subchunk1Size);
    printf("Format Type: %hd\n", header.audioFormat);
    printf("Number of Channels: %hd\n", header.numChannels);
    printf("Sample Rate: %ld\n", header.sampleRate);
    printf("Sample Rate * Bits/Sample * Channels / 8: %ld\n", header.byteRate);
    printf("Bits per Sample * Channels / 8: %hd\n", header.blockAlign);
    printf("Bits per Sample: %hd\n", header.bitsPerSample);

    //skip wExtraFormatBytes & extra format bytes
    //fseek(f, header.chunkSize - 16, SEEK_CUR);

    //Reading file
    chunk_t chunk;
    printf("id\t" "size\n");

    //go to data chunk
    while (true)
    {
        fread(&chunk, sizeof(chunk), 1, fin);
        printf("%c%c%c%c\t" "%li\n", chunk.ID[0], chunk.ID[1], chunk.ID[2], chunk.ID[3], chunk.size);
        if (*(unsigned int*)&chunk.ID == 0x61746164)
            break;
        //skip chunk data bytes
        fseek(fin, chunk.size, SEEK_CUR);
    }

    //Number of samples
    int sample_size = header.bitsPerSample / 8;
    int samples_count = chunk.size * 8 / header.bitsPerSample;
    printf("Samples count = %i\n", samples_count);
    printf(".......................................................\n");

    int* value = new int[2*samp];
    memset(value, 0, sizeof(int) * 2*samp);

    //Reading data
    for (int i = 0; i < 2*samp; i++)
    {
        fread(&value[i], 2*samp, 1, fin);
    }

    //Time Counting
    auto start = chrono::steady_clock::now();

    //fft processing
    printf("Processing FFT From CPU side\n");
    complex<double> prefetch[samp];
    complex<double> b[samp];
    for (int i = 0; i < samp; i++) {
        prefetch[i] = (double(value[i]) / 2147483648., double(value[2*i+1]) / 2147483648.);
    }

    double window[f_size];
    //hamming window
    for (int i = 0; i < samp; i += f_size / 2) {
        Hamming_Window(window, f_size);
        for (int j = 0; j < f_size && i + j < samp; j++) {
            prefetch[i + j].imag(prefetch[i + j].imag() * window[j]);
            prefetch[i + j].real(prefetch[i + j].real() * window[j]);
            //cout << b[i + j].real() << " " << b[i + j].imag() << endl;
        }
        fft(prefetch, b, 11);
    }
    FILE* fout = fopen(fileToSave, "w");
    for (int i = 0; i < samp; i++)
    {
        fprintf(fout, "%f + %fi\n", real(b[i]), imag(b[i]));
    }
    fclose(fin);
    fclose(fout);
    auto end = chrono::steady_clock::now();
    printf("Execution of CPU is Completed!\n");
    cout << "CPU Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "." << chrono::duration_cast<chrono::microseconds>(end - start).count() << "ms" << endl;
    printf(".......................................................\n");
    return value;
}

int main()
{
    //Create memory space
    IndexSave* indsave = new IndexSave[SIZE];
    thrust::complex<double>* f_ = new thrust::complex<double>[SIZE];
    int* value = new int[2*SIZE];
    memset(value, 0, sizeof(int) * 2*SIZE);
    memset(f_, 0, sizeof(thrust::complex<double>) * SIZE);

    /* CPU side*/
    value = process();

    /* GPU side*/
    printf("Processing FFT From GPU Side\n");
    GPU_kernel(value, indsave, f_);
    /*FILE* fftout = fopen("result_method1.dat", "w");
    for (int i = 0; i < samp; i++) {
        fprintf(fftout, "%f + %fi\n", f_[i].real(), f_[i].imag());
    }
    fclose(fftout);*/
}