#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mpi.h>

const int BLOCK_SIZE = 16;

std::vector<std::vector<uint8_t>> generateRandomMatrix() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    std::vector<std::vector<uint8_t>> matrix(4, std::vector<uint8_t>(4));
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            matrix[row][col] = dis(gen);
        }
    }
    return matrix;
}

void shiftRows(std::vector<std::vector<uint8_t>>& state) {
    std::rotate(state[1].begin(), state[1].begin() + 1, state[1].end());
    std::rotate(state[2].begin(), state[2].begin() + 2, state[2].end());
    std::rotate(state[3].begin(), state[3].begin() + 3, state[3].end());
}

std::vector<std::vector<std::vector<uint8_t>>> generateMatrices(size_t matrixCount) {
    std::vector<std::vector<std::vector<uint8_t>>> matrices;
    for (size_t i = 0; i < matrixCount; ++i) {
        matrices.push_back(generateRandomMatrix());
    }
    return matrices;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<size_t> dataSizes = {1 * 1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024};
    for (size_t dataSize : dataSizes) {
        size_t matrixCount = dataSize / BLOCK_SIZE;
        size_t matricesPerProc = matrixCount / size;
        size_t remainder = matrixCount % size;

        std::vector<std::vector<std::vector<uint8_t>>> matrices;
        std::vector<uint8_t> flatData;

        if (rank == 0) {
            matrices = generateMatrices(matrixCount);

            flatData.resize(matrixCount * BLOCK_SIZE);
            size_t idx = 0;
            for (const auto& matrix : matrices) {
                for (const auto& row : matrix) {
                    for (const auto& val : row) {
                        flatData[idx++] = val;
                    }
                }
            }
        }

        std::vector<int> sendCounts(size, matricesPerProc * BLOCK_SIZE);
        std::vector<int> displacements(size, 0);

        for (int i = 0; i < remainder; ++i) {
            sendCounts[i] += BLOCK_SIZE;
        }

        for (int i = 1; i < size; ++i) {
            displacements[i] = displacements[i - 1] + sendCounts[i - 1];
        }

        std::vector<uint8_t> localFlatData(sendCounts[rank]);

        MPI_Scatterv(
            flatData.data(),
            sendCounts.data(),
            displacements.data(),
            MPI_BYTE,
            localFlatData.data(),
            sendCounts[rank],
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
        );

        size_t localMatrixCount = sendCounts[rank] / BLOCK_SIZE;
        std::vector<std::vector<std::vector<uint8_t>>> localMatrices(localMatrixCount);
        size_t idx = 0;
        for (size_t i = 0; i < localMatrixCount; ++i) {
            localMatrices[i].resize(4, std::vector<uint8_t>(4));
            for (size_t row = 0; row < 4; ++row) {
                for (size_t col = 0; col < 4; ++col) {
                    localMatrices[i][row][col] = localFlatData[idx++];
                }
            }
        }

        std::vector<double> processingTimes(localMatrixCount);
        for (size_t i = 0; i < localMatrixCount; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            shiftRows(localMatrices[i]);
            auto end = std::chrono::high_resolution_clock::now();
            processingTimes[i] = std::chrono::duration<double>(end - start).count();
        }

        std::vector<int> recvCounts(size, matricesPerProc);
        std::vector<int> recvDisplacements(size, 0);

        for (int i = 0; i < remainder; ++i) {
            recvCounts[i] += 1;
        }

        for (int i = 1; i < size; ++i) {
            recvDisplacements[i] = recvDisplacements[i - 1] + recvCounts[i - 1];
        }

        std::vector<double> allProcessingTimes;
        if (rank == 0) {
            allProcessingTimes.resize(matrixCount);
        }

        MPI_Gatherv(
            processingTimes.data(),
            localMatrixCount,
            MPI_DOUBLE,
            allProcessingTimes.data(),
            recvCounts.data(),
            recvDisplacements.data(),
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0) {
            std::string timingFilename = "processing_times_" + std::to_string(dataSize / (1024 * 1024)) + "MB_" + std::to_string(size) + "procs.txt";
            std::ofstream file(timingFilename);
            for (size_t i = 0; i < allProcessingTimes.size(); ++i) {
                file << "Matrix " << i + 1 << ": " << std::fixed << std::setprecision(9) << allProcessingTimes[i] << " seconds\n";
            }
            file.close();
        }
    }

    MPI_Finalize();
    return 0;
}