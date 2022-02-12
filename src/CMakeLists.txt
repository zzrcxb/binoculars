add_executable(bino bino.c)
target_compile_options(bino PUBLIC -O1 -static)

add_executable(mont-blank mont.c)
target_compile_options(mont-blank PUBLIC -O1 -static -DBLANK)

add_executable(mont-memjam-dep mont.c)
target_compile_options(mont-memjam-dep PUBLIC -O1 -static -DMEMJAM_DEP)

add_executable(mont-memjam-para mont.c)
target_compile_options(mont-memjam-para PUBLIC -O1 -static -DMEMJAM_PARA)

add_executable(mont-bino mont.c)
target_compile_options(mont-bino PUBLIC -O1 -static)

add_executable(kaslr kaslr.c)
target_compile_options(kaslr PUBLIC -O1 -static)
