package com.example;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class MatrixMul {


    @State(Scope.Thread)
    public static class SetupState {
        INDArray nd_input = Nd4j.rand(6, 1);
        DenseMatrix jb_input = new DenseMatrix(6, 1);

        Map<Integer, INDArray> nd_w1 = new HashMap<>();
        Map<Integer, INDArray> nd_h1 = new HashMap<>();
        Map<Integer, Matrix> jb_w1 = new HashMap<>();
        Map<Integer, Matrix> jb_h1 = new HashMap<>();

        public SetupState() {
            Random random = new Random();
            int[] sizes = {50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
            for (int size : sizes) {
                nd_w1.put(size, Nd4j.rand(size, 6));
                nd_h1.put(size, Nd4j.create(size, 1));
                jb_w1.put(size, randomMtjMatrix(size, 6));
                jb_h1.put(size, new DenseMatrix(size, 1));
            }

            DenseVector input = new DenseVector(6);
            for (int i = 0; i < 6; i++) {
                input.set(i, random.nextDouble());
            }
            jb_input = new DenseMatrix(input);
        }

        private static Matrix randomMtjMatrix(final int d1,
                                              final int d2) {
            Random random = new Random();
            Matrix matrice = new DenseMatrix(d1, d2);
            for (int i = 0; i < d1; i++) {
                for (int j = 0; j < d2; j++) {
                    matrice.set(i, j, random.nextDouble());
                }
            }

            return matrice;
        }
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_mult_50(SetupState state) {
        state.nd_w1.get(50).mmuli(state.nd_input, state.nd_h1.get(50));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void jblas_mult_50(SetupState state) {
        state.jb_w1.get(50).mult(state.jb_input, state.jb_h1.get(50));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_mult_100(SetupState state) {
        state.nd_w1.get(100).mmuli(state.nd_input, state.nd_h1.get(100));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void jblas_mult_100(SetupState state) {
        state.jb_w1.get(100).mult(state.jb_input, state.jb_h1.get(100));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_mult_200(SetupState state) {
        state.nd_w1.get(200).mmuli(state.nd_input, state.nd_h1.get(200));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void jblas_mult_200(SetupState state) {
        state.jb_w1.get(200).mult(state.jb_input, state.jb_h1.get(200));
    }
    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_mult_300(SetupState state) {
        state.nd_w1.get(300).mmuli(state.nd_input, state.nd_h1.get(300));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void jblas_mult_300(SetupState state) {
        state.jb_w1.get(300).mult(state.jb_input, state.jb_h1.get(300));
    }
    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_mult_400(SetupState state) {
        state.nd_w1.get(400).mmuli(state.nd_input, state.nd_h1.get(400));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void jblas_mult_400(SetupState state) {
        state.jb_w1.get(400).mult(state.jb_input, state.jb_h1.get(400));
    }
    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_mult_500(SetupState state) {
        state.nd_w1.get(500).mmuli(state.nd_input, state.nd_h1.get(500));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void jblas_mult_500(SetupState state) {
        state.jb_w1.get(500).mult(state.jb_input, state.jb_h1.get(500));
    }
    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_mult_600(SetupState state) {
        state.nd_w1.get(600).mmuli(state.nd_input, state.nd_h1.get(600));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void jblas_mult_600(SetupState state) {
        state.jb_w1.get(600).mult(state.jb_input, state.jb_h1.get(600));
    }
    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_mult_700(SetupState state) {
        state.nd_w1.get(700).mmuli(state.nd_input, state.nd_h1.get(700));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void jblas_mult_700(SetupState state) {
        state.jb_w1.get(700).mult(state.jb_input, state.jb_h1.get(700));
    }
    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_mult_800(SetupState state) {
        state.nd_w1.get(800).mmuli(state.nd_input, state.nd_h1.get(800));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void jblas_mult_800(SetupState state) {
        state.jb_w1.get(800).mult(state.jb_input, state.jb_h1.get(800));
    }
    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_mult_900(SetupState state) {
        state.nd_w1.get(900).mmuli(state.nd_input, state.nd_h1.get(900));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void jblas_mult_900(SetupState state) {
        state.jb_w1.get(900).mult(state.jb_input, state.jb_h1.get(900));
    }
    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_mult_1000(SetupState state) {
        state.nd_w1.get(1000).mmuli(state.nd_input, state.nd_h1.get(1000));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void jblas_mult_1000(SetupState state) {
        state.jb_w1.get(1000).mult(state.jb_input, state.jb_h1.get(1000));
    }
}
