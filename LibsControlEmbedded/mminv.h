#include <stdio.h>


// Function to invert a 2x2 matrix
void inverse_2x2_matrix(float matrix[2][2], float inv_matrix[2][2]) 
{
    float det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

    // Check if the determinant is non-zero
    if (det != 0) {
        float inv_det = 1.0f / det;
        inv_matrix[0][0] = matrix[1][1] * inv_det;
        inv_matrix[0][1] = -matrix[0][1] * inv_det;
        inv_matrix[1][0] = -matrix[1][0] * inv_det;
        inv_matrix[1][1] = matrix[0][0] * inv_det;
    } else {
        printf("Matrix is singular, inverse does not exist\n");
    }
}



void inverse_4x4_matrix(float mat[4][4]) 
{
    float inv[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            inv[i][j] = 0.0;
        }
        inv[i][i] = 1.0;
    }

    for (int k = 0; k < 4; k++) {
        float factor_0 = mat[1][k] / mat[0][k];
        float factor_1 = mat[2][k] / mat[0][k];
        float factor_2 = mat[3][k] / mat[0][k];
        for (int j = k; j < 4; j++) {
            mat[1][j] -= factor_0 * mat[0][j];
            mat[2][j] -= factor_1 * mat[0][j];
            mat[3][j] -= factor_2 * mat[0][j];
            inv[1][j] -= factor_0 * inv[0][j];
            inv[2][j] -= factor_1 * inv[0][j];
            inv[3][j] -= factor_2 * inv[0][j];
        }

        float factor_3 = mat[2][k + 1] / mat[1][k + 1];
        float factor_4 = mat[3][k + 1] / mat[1][k + 1];
        for (int j = k + 1; j < 4; j++) {
            mat[2][j] -= factor_3 * mat[1][j];
            mat[3][j] -= factor_4 * mat[1][j];
            inv[2][j] -= factor_3 * inv[1][j];
            inv[3][j] -= factor_4 * inv[1][j];
        }

        float factor_5 = mat[3][k + 2] / mat[2][k + 2];
        for (int j = k + 2; j < 4; j++) {
            mat[3][j] -= factor_5 * mat[2][j];
            inv[3][j] -= factor_5 * inv[2][j];
        }
    }

    for (int i = 0; i < 4; i++) {
        float diag = mat[i][i];
        for (int j = 0; j < 4; j++) {
            mat[i][j] /= diag;
            inv[i][j] /= diag;
        }
    }
}
