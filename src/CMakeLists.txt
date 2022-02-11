add_executable(bino bino.c)
target_compile_options(bino PUBLIC -O1 -static)

add_executable(mont-blank mont.c)
target_compile_options(mont-blank PUBLIC -O1 -static -DBLANK)

add_executable(mont-memjam mont.c)
target_compile_options(mont-memjam PUBLIC -O1 -static -DMEMJAM)

add_executable(mont-memjam2 mont.c)
target_compile_options(mont-memjam2 PUBLIC -O1 -static -DMEMJAM2)

add_executable(mont-bino mont.c)
target_compile_options(mont-bino PUBLIC -O1 -static)

add_executable(kaslr kaslr.c)
target_compile_options(kaslr PUBLIC -O1 -static)
