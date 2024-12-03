#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mpi.h>

const int BLOCK_SIZE = 16;

// Функция генерации случайной матрицы 4x4
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

// Функция сдвига строк
void shiftRows(std::vector<std::vector<uint8_t>>& state) {
    std::rotate(state[1].begin(), state[1].begin() + 1, state[1].end());
    std::rotate(state[2].begin(), state[2].begin() + 2, state[2].end());
    std::rotate(state[3].begin(), state[3].begin() + 3, state[3].end());
}

// Генерация N матриц
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

        std::vector<std::vector<std::vector<uint8_t>>> matrices;
        if (rank == 0) {
            matrices = generateMatrices(matrixCount);
        }

        size_t localMatrixCount = matrixCount / size;
        std::vector<std::vector<std::vector<uint8_t>>> localMatrices(localMatrixCount);

        // Разделение матриц
        if (rank == 0) {
            for (int proc = 1; proc < size; ++proc) {
                MPI_Send(&matrices[proc * localMatrixCount], localMatrixCount * BLOCK_SIZE, MPI_BYTE, proc, 0, MPI_COMM_WORLD);
            }
            localMatrices = std::vector<std::vector<std::vector<uint8_t>>>(matrices.begin(), matrices.begin() + localMatrixCount);
        } else {
            localMatrices.resize(localMatrixCount);
            MPI_Recv(&localMatrices[0], localMatrixCount * BLOCK_SIZE, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Обработка локальных матриц
        std::vector<double> processingTimes(localMatrixCount);
        for (size_t i = 0; i < localMatrixCount; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            shiftRows(localMatrices[i]);
            auto end = std::chrono::high_resolution_clock::now();
            processingTimes[i] = std::chrono::duration<double>(end - start).count();
        }

        // Сбор результатов
        if (rank == 0) {
            std::vector<double> allProcessingTimes(matrixCount);
            for (int proc = 1; proc < size; ++proc) {
                MPI_Recv(&allProcessingTimes[proc * localMatrixCount], localMatrixCount, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            std::copy(processingTimes.begin(), processingTimes.end(), allProcessingTimes.begin());

            // Сохранение времени обработки
            std::string timingFilename = "processing_times_" + std::to_string(dataSize / (1024 * 1024)) + "MB_" + std::to_string(size) + "procs.txt";
            std::ofstream file(timingFilename);
            for (size_t i = 0; i < allProcessingTimes.size(); ++i) {
                file << "Matrix " << i + 1 << ": " << std::fixed << std::setprecision(9) << allProcessingTimes[i] << " seconds\n";
            }
            file.close();
        } else {
            MPI_Send(&processingTimes[0], localMatrixCount, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
