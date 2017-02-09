	.file	"thread.c"
	.comm	mylock,40,32
	.comm	mylock2,40,32
	.comm	mylock3,40,32
	.comm	mylock4,40,32
	.comm	mylock5,40,32
	.comm	total_hits,8,8
	.comm	num_iter,4,4
	.comm	hits,8,8
	.text
.globl foo
	.type	foo, @function
foo:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -88(%rbp)
	movq	-88(%rbp), %rax
	movq	%rax, -56(%rbp)
	leaq	-80(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	gettimeofday
	call	gsl_rng_env_setup
	movq	gsl_rng_default(%rip), %rax
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	gsl_rng_alloc
	movq	%rax, -32(%rbp)
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	-80(%rbp), %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	leaq	(%rdx,%rax), %rax
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	gsl_rng_set
	movl	$0, -44(%rbp)
	movl	$0, -48(%rbp)
	jmp	.L2
.L4:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	gsl_rng_uniform
	movsd	.LC0(%rip), %xmm1
	subsd	%xmm1, %xmm0
	addsd	%xmm0, %xmm0
	movsd	%xmm0, -24(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	gsl_rng_uniform
	movsd	.LC0(%rip), %xmm1
	subsd	%xmm1, %xmm0
	addsd	%xmm0, %xmm0
	movsd	%xmm0, -16(%rbp)
	movsd	-24(%rbp), %xmm0
	movapd	%xmm0, %xmm1
	mulsd	-24(%rbp), %xmm1
	movsd	-16(%rbp), %xmm0
	mulsd	-16(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -8(%rbp)
	movsd	.LC1(%rip), %xmm0
	ucomisd	-8(%rbp), %xmm0
	seta	%al
	testb	%al, %al
	je	.L3
	addl	$1, -44(%rbp)
.L3:
	addl	$1, -48(%rbp)
.L2:
	movl	num_iter(%rip), %eax
	cmpl	%eax, -48(%rbp)
	jl	.L4
	movq	hits(%rip), %rdx
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	-44(%rbp), %eax
	movl	%eax, (%rdx)
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	foo, .-foo
	.section	.rodata
.LC3:
	.string	"PI = %2.10f\n"
.LC4:
	.string	"(real) PI = %2.10f\n"
	.text
.globl main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	$0, %esi
	movl	$mylock, %edi
	call	pthread_mutex_init
	movq	$0, total_hits(%rip)
	movl	$40, -4(%rbp)
	movl	$1000000, num_iter(%rip)
	movl	-4(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc
	movq	%rax, -24(%rbp)
	movl	-4(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc
	movq	%rax, -16(%rbp)
	movl	-4(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc
	movq	%rax, hits(%rip)
	movl	$0, -8(%rbp)
	jmp	.L7
.L8:
	movl	-8(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	-16(%rbp), %rax
	movl	-8(%rbp), %edx
	movl	%edx, (%rax)
	movl	-8(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdx
	addq	-16(%rbp), %rdx
	movl	-8(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	-24(%rbp), %rax
	movq	%rdx, %rcx
	movl	$foo, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create
	addl	$1, -8(%rbp)
.L7:
	movl	-8(%rbp), %eax
	cmpl	-4(%rbp), %eax
	jl	.L8
	movl	$0, -8(%rbp)
	jmp	.L9
.L10:
	movl	-8(%rbp), %eax
	cltq
	salq	$3, %rax
	addq	-24(%rbp), %rax
	movq	(%rax), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_join
	addl	$1, -8(%rbp)
.L9:
	movl	-8(%rbp), %eax
	cmpl	-4(%rbp), %eax
	jl	.L10
	movq	$0, total_hits(%rip)
	movl	$0, -8(%rbp)
	jmp	.L11
.L12:
	movq	hits(%rip), %rax
	movl	-8(%rbp), %edx
	movslq	%edx, %rdx
	salq	$2, %rdx
	addq	%rdx, %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	total_hits(%rip), %rax
	leaq	(%rdx,%rax), %rax
	movq	%rax, total_hits(%rip)
	addl	$1, -8(%rbp)
.L11:
	movl	-8(%rbp), %eax
	cmpl	-4(%rbp), %eax
	jl	.L12
	movq	total_hits(%rip), %rax
	cvtsi2sdq	%rax, %xmm0
	movsd	.LC2(%rip), %xmm1
	mulsd	%xmm1, %xmm0
	movl	num_iter(%rip), %eax
	imull	-4(%rbp), %eax
	cvtsi2sd	%eax, %xmm1
	divsd	%xmm1, %xmm0
	movl	$.LC3, %eax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf
	movl	$.LC4, %eax
	movsd	.LC5(%rip), %xmm0
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC0:
	.long	0
	.long	1071644672
	.align 8
.LC1:
	.long	0
	.long	1072693248
	.align 8
.LC2:
	.long	0
	.long	1074790400
	.align 8
.LC5:
	.long	1413754136
	.long	1074340347
	.ident	"GCC: (GNU) 4.4.7 20120313 (Red Hat 4.4.7-17)"
	.section	.note.GNU-stack,"",@progbits
