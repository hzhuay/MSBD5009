#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main() {
#define MASTER 0

	MPI_Init(NULL, NULL);

	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int sumnum = 15;
	short *sgn = (short *)malloc(sumnum * sizeof(short));

	if (world_rank == MASTER) {
		printf("Sum data at rank %d = {", world_rank);
		for (int i = 0; i < sumnum; i++) {
			sgn[i] = i + 1;
			if (i == sumnum - 1) {
				printf("%d}\n", sgn[i]);
			} else {
				printf("%d, ", sgn[i]);
			}
		}
	}

	const int localnum = world_rank + 3;
	int *localnums = NULL, *offsets = NULL;

	short *local_sgn = (short*)malloc(sizeof(short)*localnum);

	if (world_rank == MASTER) {
		localnums = (int *)malloc(world_size * sizeof(int));
		offsets = (int *)malloc((world_size + 1) * sizeof(int));
	}
	// everyone contributes their info
	MPI_Gather(&localnum, 1, MPI_INT, localnums, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	// the root constructs the offsets array
	if (world_rank == MASTER) {
		offsets[0] = 0;
		for (int i = 0; i < world_size; i++) {
			offsets[i + 1] = offsets[i] + localnums[i] - 1; // Consider the overlap
		}

		if ((sumnum - 1) != offsets[world_size]) {
			fprintf(stderr, "The sum of nodes is %d, but the root "
				"process only scattered %d nodes' data.\n",
				(sumnum - 1), offsets[world_size]);
		}
	}
	// everyone contributes their data
	MPI_Scatterv(sgn, localnums, offsets, MPI_SHORT,
		local_sgn, localnum, MPI_SHORT,
		MASTER, MPI_COMM_WORLD);
	free(localnums);
	free(offsets);
	free(sgn);

	printf("Scattered data at rank %d = {", world_rank);
	for (int i = 0; i < localnum; i++) {
		if (i == localnum - 1) {
			printf("%d}\n", local_sgn[i]);
		} else {
			printf("%d, ", local_sgn[i]);
		}
	}

	free(local_sgn);
	MPI_Finalize();
	return 0;
}
