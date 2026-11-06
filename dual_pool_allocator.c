/*
 * dual_pool_allocator.c
 *
 * Dual-Pool Based KV Cache Memory Allocator for VLM Visual Token Management
 * Data Structures Course Project
 *
 * Components:
 *   - SmallSlabPool  : Fixed-size slab pool for text tokens (Free List)
 *   - LargeBlockPool : Variable-size block pool for visual tokens (Best-Fit)
 *   - DualPoolAllocator : Unified interface routing requests to the right pool
 *   - main()         : Benchmark simulation vs. malloc
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#define SMALL_BLOCK_SIZE  256
#define SMALL_POOL_COUNT  8192
#define LARGE_POOL_BYTES  (64 * 1024 * 1024) // 64MB
#define SMALL_THRESHOLD   256


//  1. SmallSlabPool - Fixed-size Slab Pool (Text Tokens)


typedef struct FreeNode 
{
    struct FreeNode *next;
} FreeNode;

typedef struct 
{
    char *pool_base;
    FreeNode *free_head;
    size_t total_slots;
    size_t used_slots;
} SmallSlabPool;

void SmallSlabPool_init(SmallSlabPool *pool, size_t slot_count) 
{
    pool->total_slots = slot_count;
    pool->used_slots = 0;
    pool->pool_base = (char* ) malloc(slot_count * SMALL_BLOCK_SIZE);
    if (!pool->pool_base) 
    {
        fprintf(stderr, "[SmallSlabPool] Init failed\n");
        exit(1);
    }

    pool->free_head = NULL;
    for (size_t i = slot_count; i > 0; i--) 
    {
        char *slot = pool->pool_base + (i - 1) * SMALL_BLOCK_SIZE;
        FreeNode *node = (FreeNode *) slot;
        node->next = pool->free_head;
        pool->free_head = node;
    }
}

void SmallSlabPool_destroy(SmallSlabPool *pool) 
{
    free(pool->pool_base);
}

void *SmallSlabPool_allocate(SmallSlabPool *pool) 
{
    if (!pool->free_head) return NULL;
    void *slot = (void *) pool->free_head;
    pool->free_head = pool->free_head->next;
    pool->used_slots++;
    return slot;
}

void SmallSlabPool_deallocate(SmallSlabPool *pool, void *ptr) 
{
    if (!ptr) return;
    FreeNode *node = (FreeNode *) ptr;
    node->next = pool->free_head;
    pool->free_head = node;
    pool->used_slots--;
}


//  2. LargeBlockPool - Variable-size Block Pool (Visual Tokens)

typedef struct BlockHeader 
{
    size_t size;
    bool is_free;
    struct BlockHeader *next;
} BlockHeader;

#define HEADER_SIZE sizeof(BlockHeader)

typedef struct 
{
    char *pool_base;
    BlockHeader *head;
    size_t total_bytes;
} LargeBlockPool;

void LargeBlockPool_init(LargeBlockPool *pool, size_t pool_size) 
{
    pool->total_bytes = pool_size;
    pool->pool_base = (char *) malloc(pool_size);
    if (!pool->pool_base) 
    {
        fprintf(stderr, "[LargeBlockPool] Init failed\n");
        exit(1);
    }

    pool->head = (BlockHeader *) pool->pool_base;
    pool->head->size = pool_size - HEADER_SIZE;
    pool->head->is_free = true;
    pool->head->next = NULL;
}

void LargeBlockPool_destroy(LargeBlockPool *pool)
{
    free(pool->pool_base);
}

void *LargeBlockPool_allocate(LargeBlockPool *pool, size_t size) 
{
    if (size == 0) return NULL;

    BlockHeader *best = NULL;
    size_t best_diff = SIZE_MAX;

    for (BlockHeader *cur = pool->head; cur; cur = cur->next) 
    {
        if (cur->is_free && cur->size >= size) 
        {
            size_t diff = cur->size - size;
            if (diff < best_diff) 
            {
                best_diff = diff;
                best = cur;
                if (diff == 0) break; // perfect fit
            }
        }
    }

    if (!best) return NULL;

    // Split if leftover space is worth keeping
    if (best->size >= size + HEADER_SIZE + 8) 
    {
        char *split_addr = (char *) best + HEADER_SIZE + size;
        BlockHeader *sp = (BlockHeader *) split_addr;
        sp->size = best->size - size - HEADER_SIZE;
        sp->is_free = true;
        sp->next = best->next;
        best->size = size;
        best->next = sp;
    }

    best->is_free = false;
    return (char *) best + HEADER_SIZE;
}

void LargeBlockPool_deallocate(LargeBlockPool *pool, void *ptr) 
{
    if (!ptr) return;
    BlockHeader *target = (BlockHeader *)((char *) ptr - HEADER_SIZE);
    target->is_free = true;

    for (BlockHeader *cur = pool->head; cur; cur = cur->next) 
    {
        while (cur->is_free && cur->next && cur->next->is_free) 
        {
            cur->size += HEADER_SIZE + cur->next->size;
            cur->next = cur->next->next;
        }
    }
}

double LargeBlockPool_fragmentation_ratio(const LargeBlockPool *pool) 
{
    size_t total_free = 0, largest_free = 0;
    for (BlockHeader *cur = pool->head; cur; cur = cur->next) 
    {
        if (cur->is_free) 
        {
            total_free += cur->size;
            if (cur->size > largest_free) largest_free = cur->size;
        }
    }
    if (total_free == 0) return 0.0;
    return 1.0 - ((double)largest_free / (double)total_free);
}

size_t LargeBlockPool_get_used_bytes(const LargeBlockPool *pool) 
{
    size_t used = 0;
    for (BlockHeader *cur = pool->head; cur; cur = cur->next) 
    {
        if (!cur->is_free) used += cur->size;
    }
    return used;
}


//  3. DualPoolAllocator - Unified Interface

typedef struct 
{
    SmallSlabPool  small_pool;
    LargeBlockPool large_pool;

    size_t alloc_count_small;
    size_t alloc_count_large;
    size_t fail_count;
} DualPoolAllocator;

void DualPoolAllocator_init(DualPoolAllocator *alloc) 
{
    SmallSlabPool_init(&alloc->small_pool, SMALL_POOL_COUNT);
    LargeBlockPool_init(&alloc->large_pool, LARGE_POOL_BYTES);
    alloc->alloc_count_small = 0;
    alloc->alloc_count_large = 0;
    alloc->fail_count = 0;
}

void DualPoolAllocator_destroy(DualPoolAllocator *alloc) 
{
    SmallSlabPool_destroy(&alloc->small_pool);
    LargeBlockPool_destroy(&alloc->large_pool);
}

void *DualPoolAllocator_allocate(DualPoolAllocator *alloc, size_t size) 
{
    void *ptr = NULL;
    if (size <= SMALL_THRESHOLD) 
    {
        ptr = SmallSlabPool_allocate(&alloc->small_pool);
        if (ptr) alloc->alloc_count_small++;
    } else 
    {
        ptr = LargeBlockPool_allocate(&alloc->large_pool, size);
        if (ptr) alloc->alloc_count_large++;
    }
    if (!ptr) alloc->fail_count++;
    return ptr;
}

void DualPoolAllocator_deallocate(DualPoolAllocator *alloc, void *ptr, size_t size) 
{
    if (!ptr) return;
    if (size <= SMALL_THRESHOLD) SmallSlabPool_deallocate(&alloc->small_pool, ptr);
    else LargeBlockPool_deallocate(&alloc->large_pool, ptr);
}

void DualPoolAllocator_print_stats(const DualPoolAllocator *alloc) 
{
    printf("\n===== DualPoolAllocator Statistics =====\n");
    printf("[Small Slab Pool]\n");
    printf("  Alloc calls  : %zu\n", alloc->alloc_count_small);
    printf("  Used slots   : %zu / %zu\n", alloc->small_pool.used_slots, alloc->small_pool.total_slots);
    printf("  Memory used  : %.2f KB / %.2f KB\n",
           (alloc->small_pool.used_slots * SMALL_BLOCK_SIZE) / 1024.0,
           (alloc->small_pool.total_slots * SMALL_BLOCK_SIZE) / 1024.0);
    printf("[Large Block Pool]\n");
    printf("  Alloc calls  : %zu\n", alloc->alloc_count_large);
    printf("  Memory used  : %.2f KB / %.2f KB\n",
           LargeBlockPool_get_used_bytes(&alloc->large_pool) / 1024.0,
           alloc->large_pool.total_bytes / 1024.0);
    printf("  Frag. ratio  : %.2f %%\n", LargeBlockPool_fragmentation_ratio(&alloc->large_pool) * 100.0);
    printf("[Overall]\n");
    printf("  Fail count   : %zu\n", alloc->fail_count);
    printf("========================================\n");
}


//  4. Simulation - DualPool vs. malloc Benchmark

#define BATCH_SIZE 200
#define ROUNDS 250

typedef struct 
{
    size_t size;
    bool is_visual;
} TokenRequest;

typedef struct 
{
    void *ptr;
    size_t size;
} LiveToken;

// Simple pseudo-random generator replacing POSIX rand_r for wide C compatibility
static unsigned int custom_rand(unsigned int *seed) 
{
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return *seed;
}

void make_batch(unsigned int *seed, TokenRequest *batch) 
{
    for (int i = 0; i < BATCH_SIZE; i++) 
    {
        bool vis = (custom_rand(seed) % 5 == 0);
        size_t size;
        if (vis) 
        {
            size_t patches = 196 + custom_rand(seed) % 829;
            size = patches * 64;
        } else 
        {
            size = 32 + (size_t)(custom_rand(seed) % 8) * 32;
        }
        batch[i].size = size;
        batch[i].is_visual = vis;
    }
}

double run_dual_pool(double *out_frag_pct) 
{
    DualPoolAllocator allocator;
    DualPoolAllocator_init(&allocator);
    unsigned int seed = 42;

    size_t total_text = 0, total_visual = 0, total_fail = 0;
    TokenRequest batch[BATCH_SIZE];
    LiveToken live[BATCH_SIZE];

    clock_t t0 = clock();

    for (int r = 0; r < ROUNDS; r++) 
    {
        make_batch(&seed, batch);
        size_t live_count = 0;

        // ALLOC phase
        for (int i = 0; i < BATCH_SIZE; i++) 
        {
            void *p = DualPoolAllocator_allocate(&allocator, batch[i].size);
            if (p) 
            {
                live[live_count].ptr = p;
                live[live_count].size = batch[i].size;
                live_count++;
                if (batch[i].is_visual) total_visual++;
                else total_text++;
            } else 
            {
                total_fail++;
            }
        }

        // FREE phase
        for (size_t i = 0; i < live_count; i++) 
        {
            DualPoolAllocator_deallocate(&allocator, live[i].ptr, live[i].size);
        }
    }

    clock_t t1 = clock();

    DualPoolAllocator_print_stats(&allocator);
    if (out_frag_pct) *out_frag_pct = LargeBlockPool_fragmentation_ratio(&allocator.large_pool) * 100.0;

    printf("  Text allocs   : %zu\n", total_text);
    printf("  Visual allocs : %zu\n", total_visual);
    printf("  Failures      : %zu\n", total_fail);

    DualPoolAllocator_destroy(&allocator);
    return (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC;
}

double run_malloc(void) 
{
    unsigned int seed = 42;
    TokenRequest batch[BATCH_SIZE];
    void *live[BATCH_SIZE];

    clock_t t0 = clock();

    for (int r = 0; r < ROUNDS; r++) 
    {
        make_batch(&seed, batch);
        size_t live_count = 0;

        for (int i = 0; i < BATCH_SIZE; i++) 
        {
            void *p = malloc(batch[i].size);
            if (p) 
            {
                live[live_count++] = p;
            }
        }
        for (size_t i = 0; i < live_count; i++) 
        {
            free(live[i]);
        }
    }

    clock_t t1 = clock();
    return (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC;
}

int main(void) 
{
    const int TOTAL_OPS = ROUNDS * BATCH_SIZE;

    printf("================================================\n");
    printf("  VLM Dual-Pool Allocator — Performance Benchmark\n");
    printf("================================================\n");
    printf("Rounds         : %d\n", ROUNDS);
    printf("Batch size     : %d tokens/round\n", BATCH_SIZE);
    printf("Total ops      : %d alloc+free pairs\n", TOTAL_OPS);
    printf("Visual tokens  : ~20%%  (12KB – 65KB)\n");
    printf("Text tokens    : ~80%%  (32B  – 256B)\n\n");

    // DualPool
    printf("--- Running DualPoolAllocator ---\n");
    double frag_pct = 0.0;
    double dual_ms = run_dual_pool(&frag_pct);

    // malloc
    printf("\n--- Running malloc ---\n");
    double malloc_ms = run_malloc();
    printf("  (no per-op stats for malloc)\n");

    // Summary
    printf("\n================================================\n");
    printf("             BENCHMARK RESULTS\n");
    printf("================================================\n");
    printf("DualPoolAllocator : %10.3f ms\n", dual_ms);
    printf("malloc            : %10.3f ms\n", malloc_ms);
    printf("Speedup           : %10.3fx\n", malloc_ms / dual_ms);
    printf("Large Pool Frag.  : %.2f %%\n", frag_pct);
    printf("================================================\n");

    return 0;
}
