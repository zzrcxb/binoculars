#pragma once

#if defined(HASWELL)
#define EV_SIZE 12
#define STEP_BIT 31
#define CENTRAL_ZERO 0x930
// HASWELL

#elif defined(SKYLAKE)
#define EV_SIZE 24
#define STEP_BIT 31
#define CENTRAL_ZERO 0x8d0
// SKYLAKE

#elif defined(CASCADE)
#define EV_SIZE 24
#define STEP_BIT 31
#define CENTRAL_ZERO 0x8d0
// CASCADE

#endif

#ifdef MEMJAM_DEP
#undef EV_SIZE
#define EV_SIZE 2
#endif // MEMJAM_DEP

#ifdef MEMJAM_PARA
#undef EV_SIZE
#define EV_SIZE 8
#undef STEP_BIT
#define STEP_BIT 12
#endif // MEMJAM_PARA

#ifdef BLANK
#undef EV_SIZE
#define EV_SIZE 2
#endif // BLANK
