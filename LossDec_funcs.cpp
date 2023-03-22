# include "LossDec_funcs.h"

# include <iostream>
# include <stdlib.h>
# include <math.h>
# include <random>
# include <functional>
# include <cstdlib>  // rand()
# include <cstring>  // memcpy

void LossDecoder_GaussElimin(data_type* mat, data_type* dummy, int n_rows, int n_cols, int num_lost_qbts) {
	int ix_r, lead, i;
	data_type* m;

	lead = 0;

	for (ix_r = 0; ix_r < (n_rows - 1); ix_r++) {

		if (lead >= num_lost_qbts) {
			return;
		}

		m = mat + lead + n_cols * ix_r;

		i = ix_r;
		while (*m == 0)
		{
			++i;
			if (i == (n_rows - 1)) {
				i = ix_r;
				++lead;
				if (lead == num_lost_qbts) {
					return;
				}
				m = mat + lead + n_cols * ix_r;
			}
			else {
				m += n_cols;
			}

		}


		if (i != ix_r) {
			SwitchRowsBlock(mat, dummy, i, ix_r, n_rows, n_cols);
		}

		m = mat + lead + (ix_r + 1) * n_cols;
		for (i = ix_r + 1; i < n_rows; i++) {
			if (*m != 0) {
				SubtractRows(mat, ix_r, i, n_rows, n_cols);
			}
			m += n_cols;
		}

		lead++;

	}
	return;
}


void LossDecoder_GaussElimin_print(data_type* mat, int n_rows, int n_cols, int num_lost_qbts) {
	int ix_r, lead, i;
	data_type* m;

	lead = 0;

	for (ix_r = 0; ix_r < (n_rows - 1); ix_r++) {
        std::cout << "\nStarting row " << ix_r << std::endl;
        PrintMatrix_toTerminal(mat, n_rows, n_cols);

		if (lead >= num_lost_qbts) {
			return;
		}

		m = mat + lead + n_cols * ix_r;

		i = ix_r;
		while (*m == 0)
		{
			i = i+1;
			if (i == (n_rows - 1)) {
				i = ix_r;
				lead = lead + 1;
				if (lead == num_lost_qbts) {
					return;
				}
				m = mat + lead + n_cols * ix_r;
			}
			else {
				m += n_cols;
			}

		}

        std::cout << "\nFound lead in row " << i << "  col " << lead << std::endl;

		if (i != ix_r) {
            std::cout << "\nSwitching rows " << ix_r << " and "<< i << std::endl;
			SwitchRows(mat, i, ix_r, n_rows, n_cols);
            PrintMatrix_toTerminal(mat, n_rows, n_cols);
		}

		m = mat + lead + (ix_r + 1) * n_cols;
		for (i = ix_r + 1; i < n_rows; i++) {
			if (*m != 0) {
                std::cout << "\nSumming row " << i << " to row " << ix_r << std::endl;
				SubtractRows(mat, ix_r, i, n_rows, n_cols);
                PrintMatrix_toTerminal(mat, n_rows, n_cols);
			}
			m += n_cols;
		}

		lead++;

	}
	return;
}

void SwitchRows(data_type* mat, int row0, int row1, int n_rows, int n_cols) {
	int i;
	data_type* p0, * p1, temp;

	p0 = mat + row0 * n_cols;
	p1 = mat + row1 * n_cols;

	for (i = 0; i < n_cols; i++) {
		temp = *p0;
		*p0 = *p1;
		*p1 = temp;
		++p0;
		++p1;
	}
}

void SwitchRowsBlock(data_type* mat, data_type* dummy, int row0, int row1, int n_rows, int n_cols) {
	data_type* p0 = mat + row0 * n_cols;
	data_type* p1 = mat + row1 * n_cols;
	memcpy(dummy, p0, sizeof(data_type) * n_cols);
	memcpy(p0, p1, sizeof(data_type) * n_cols);
	memcpy(p1, dummy, sizeof(data_type) * n_cols);
}

void SubtractRows(data_type* mat, int row0, int row_target, int n_rows, int n_cols) {
	int i;
	data_type* p0, * ptarg;

	p0 = mat + row0 * n_cols;
	ptarg = mat + row_target * n_cols;

	for (i = 0; i < n_cols; i++) {
		*ptarg = *ptarg ^ *p0;
		++p0;
		++ptarg;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void LossDecoder_GaussElimin_trackqbts(data_type* mat, int* qbt_syndr_mat, int n_rows, int n_cols, int num_lost_qbts) {
	int ix_r, lead, i, *track_synds_mat;
	data_type* m;

    // std::cout << "\nStarting row LossDecoder_GaussElimin_trackqbts" << std::endl;


	lead = 0;
	//////////////////////////////////
	// PrintMatrix_int_toTerminal(qbt_syndr_mat, n_cols, 2);
	//////////////////////////////////

	for (ix_r = 0; ix_r < (n_rows - 1); ix_r++) {

		if (lead >= num_lost_qbts) {
			return;
		}

		m = mat + lead + n_cols * ix_r;

		i = ix_r;
		while (*m == 0)
		{
			++i;
			if (i == (n_rows - 1)) {
				i = ix_r;
				++lead;
				if (lead == num_lost_qbts) {
					return;
				}
				m = mat + lead + n_cols * ix_r;
			}
			else {
				m += n_cols;
			}

		}


		if (i != ix_r) {
			// std::cout << "\nSwitching rows " << ix_r << " and "<< i << std::endl;
			SwitchRows_trackqbts(mat, qbt_syndr_mat, i, ix_r, n_rows, n_cols);
		}

		m = mat + lead + (ix_r + 1) * n_cols;
		for (i = ix_r + 1; i < n_rows; i++) {
			if (*m != 0) {
				SubtractRows_trackqbts(mat, qbt_syndr_mat, ix_r, i, n_rows, n_cols);
			}
			m += n_cols;
		}

		lead++;

	}
	return;
}


void SwitchRows_trackqbts(data_type* mat, int* qbt_syndr_mat, int row0, int row1, int n_rows, int n_cols) {
	bool check_order=false;
	int i, *track_synds_mat, temp_synd;
	data_type* p0, * p1, temp;

	p0 = mat + row0 * n_cols;
	p1 = mat + row1 * n_cols;

	track_synds_mat = qbt_syndr_mat;
	// PrintMatrix_toTerminal(mat, n_rows, n_cols);
	// PrintMatrix_int_toTerminal(qbt_syndr_mat, n_cols, 2);
	for (i = 0; i < n_cols; i++) {
		temp = *p0;
		*p0 = *p1;
		*p1 = temp;
		if (*p0 || *p1){
			// std::cout << "\nUpdating syndrs for qubit "<< i << " with synds " << *track_synds_mat << " and " << *(track_synds_mat+1) << std::endl;
			// Track new syndromes for this qubit
			if (*track_synds_mat == row0){
				*track_synds_mat = row1;
				check_order=true;
			} else if (*track_synds_mat == row1){
				*track_synds_mat = row0;
				check_order=true;
			}
			++track_synds_mat;
			if (*track_synds_mat == row0){
				*track_synds_mat = row1;
				check_order=true;
			} else if (*track_synds_mat == row1){
				*track_synds_mat = row0;
				check_order=true;
			}
			// Maintain increasing order of syndromes for each qubit in qbt_syndr_mat
			if (check_order && (*track_synds_mat < *(track_synds_mat-1))){
				temp_synd = *track_synds_mat;
				*track_synds_mat = *(track_synds_mat-1);
				*(track_synds_mat-1) = temp_synd;
				check_order=false;
			}
			// PrintMatrix_int_toTerminal(qbt_syndr_mat, n_cols, 2);
			++track_synds_mat;
		} else {
			track_synds_mat += 2;
		}
		++p0;
		++p1;
	}
}

void SubtractRows_trackqbts(data_type* mat, int* qbt_syndr_mat, int row0, int row_target, int n_rows, int n_cols) {
	bool check_order=false;
	int i, *track_synds_mat, temp_synd;
	data_type* p0, * ptarg;

	p0 = mat + row0 * n_cols;
	ptarg = mat + row_target * n_cols;


		// Track new syndromes for this qubit



	if (row_target < (n_rows-1)){  // if the row is the last one, i.e. the logical operator, no need to track the qubit syndromes
		track_synds_mat = qbt_syndr_mat;
		for (i = 0; i < n_cols; i++) {
			*ptarg = *ptarg ^ *p0;

			if (*p0){
				if (*ptarg){
					if (*track_synds_mat == row0){
						*track_synds_mat = row_target;
						check_order=true;
					}
					++track_synds_mat;
					if (*track_synds_mat == row0){
						*track_synds_mat = row_target;
						check_order=true;
					}
					if (check_order && (*track_synds_mat < *(track_synds_mat-1))){
						temp_synd = *track_synds_mat;
						*track_synds_mat = *(track_synds_mat-1);
						*(track_synds_mat-1) = temp_synd;
						check_order=false;
					}
				} else {
					if (*track_synds_mat == row_target){
						*track_synds_mat = row0;
						check_order=true;
					}
					++track_synds_mat;
					if (*track_synds_mat == row_target){
						*track_synds_mat = row0;
						check_order=true;
					}
				}
				++track_synds_mat;
			} else {
				track_synds_mat += 2;
			}
			++p0;
			++ptarg;
		}		
	} else {
		for (i = 0; i < n_cols; i++) {
			*ptarg = *ptarg ^ *p0;
			++p0;
			++ptarg;
		}
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void PrintMatrix_toTerminal(data_type* mat, int n_rows, int n_cols) {
	data_type* m;
	int ix_r, ix_c;

	m = mat;

	for (ix_r = 0; ix_r < n_rows; ix_r++)
	{
		std::cout << "\n";
		for (ix_c = 0; ix_c < n_cols; ix_c++)
		{
			std::cout << *m << " ";
			++m;
		}
	}
}


void PrintMatrix_int_toTerminal(int* mat, int n_rows, int n_cols) {
	int* m;
	int ix_r, ix_c;

	m = mat;

	for (ix_r = 0; ix_r < n_rows; ix_r++)
	{
		std::cout << "\n";
		for (ix_c = 0; ix_c < n_cols; ix_c++)
		{
			std::cout << *m << " ";
			++m;
		}
	}
	std::cout << std::endl;
}


void random_maxtrix(data_type* mat, int n_rows, int n_cols){
  for(int r=0; r<n_rows; r++){
    for(int c=0; c<n_cols; c++){
      mat[r*n_cols + c] = rand() % 2;
    }
  }
}

void print_bitwise(data_type_b* row, uint64_t num_blocks){
  for(uint64_t i=0; i<num_blocks; i++){
    for(uint8_t byte=0; byte<sizeof(data_type_b); byte++){
      for(uint8_t bit=0; bit<8; bit++){
        printf("%i", (row[i] >> (byte*8+bit)) & 1ul ? 1 : 0);
      }
      printf(",");
    }
    printf("|");
  }
  printf("\n");
}

int main(int argc , char** argv) {
  int ans;
  uint64_t cnt = 0;
  if (argc != 2) {
    fprintf(stderr, "Invalid number of arguments.\n");
    return 1;
  }
  int nQubit = 10;
  int n_rows = nQubit;
  int n_cols = nQubit * 2;
  int num_lost_qbts = nQubit;

  data_type* mat = (data_type*)malloc(sizeof(data_type) * n_rows * n_cols);
  data_type* dummy = (data_type*)malloc(sizeof(data_type) * n_cols); // dummy memory for swapping rows

  random_maxtrix(mat, n_rows, n_cols);
  PrintMatrix_toTerminal(mat, n_rows, n_cols);
  LossDecoder_GaussElimin(mat, dummy, n_rows, n_cols, num_lost_qbts);
  printf("%s\n", "");
  PrintMatrix_toTerminal(mat, n_rows, n_cols);
  printf("%s\n", "");

  // some tests for bit-wise operations
  n_cols = 100;
  uint64_t num_blocks = n_cols / (8 * sizeof(data_type_b)) + 1;
  uint64_t num_bytes = sizeof(data_type_b) * num_blocks;
  data_type_b* testRow0 = (data_type_b*)malloc(num_bytes);
  data_type_b* testRow1 = (data_type_b*)malloc(num_bytes);
  data_type_b* testRow2 = (data_type_b*)malloc(num_bytes);
  memset(testRow0, 250, num_bytes);
  memset(testRow1, 7, num_bytes);

  printf("bitwise or:\n");
  for(uint64_t i=0; i<num_blocks; i++){
    testRow2[i] = testRow0[i] | testRow1[i];
  }
  print_bitwise(testRow0, num_blocks);
  print_bitwise(testRow1, num_blocks);
  print_bitwise(testRow2, num_blocks);

  printf("bitwise and:\n");
  for(uint64_t i=0; i<num_blocks; i++){
    testRow2[i] = testRow0[i] & testRow1[i];
  }
  print_bitwise(testRow0, num_blocks);
  print_bitwise(testRow1, num_blocks);
  print_bitwise(testRow2, num_blocks);

  printf("bitwise xor:\n");
  for(uint64_t i=0; i<num_blocks; i++){
    testRow2[i] = testRow0[i] ^ testRow1[i];
  }
  print_bitwise(testRow0, num_blocks);
  print_bitwise(testRow1, num_blocks);
  print_bitwise(testRow2, num_blocks);

  free(testRow0);free(testRow1);free(testRow2);

  free(mat);
  return 0;
}
