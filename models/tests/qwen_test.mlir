module {
  func.func @main(%arg0: tensor<64x16xf32>, %arg1: tensor<64x16xf32>, %arg2: tensor<64x16xf32>, %arg3: tensor<64x16xf32>, %arg4: tensor<256x128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<384x128xf32>, %arg7: tensor<128x128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<512x128xf32>, %arg10: tensor<512x128xf32>, %arg11: tensor<128x512xf32>, %arg12: tensor<128xf32>, %arg13: tensor<384x128xf32>, %arg14: tensor<128x128xf32>, %arg15: tensor<128xf32>, %arg16: tensor<512x128xf32>, %arg17: tensor<512x128xf32>, %arg18: tensor<128x512xf32>, %arg19: tensor<128xf32>, %arg20: tensor<64x16xf32>, %arg21: tensor<64x16xf32>, %arg22: tensor<8x64xi64>, %arg23: tensor<8x64xi64>) -> tensor<f32> {
    %c = stablehlo.constant dense<0> : tensor<512xi64>
    %c_0 = stablehlo.constant dense<-100> : tensor<512xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<64x64xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<64x64xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf32>
    %cst_6 = arith.constant dense<2> : tensor<1xi64>
    %cst_7 = arith.constant dense<128> : tensor<1xi64>
    %cst_8 = arith.constant dense<9.9999999999999995E-7> : tensor<1xf64>
    %cst_9 = arith.constant dense<0.17677669529663687> : tensor<1xf64>
    %0 = "stablehlo.gather"(%arg4, %arg22) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 128>}> : (tensor<256x128xf32>, tensor<8x64xi64>) -> tensor<8x64x128xf32>
    %1 = stablehlo.convert %0 : tensor<8x64x128xf32>
    %2 = stablehlo.convert %cst_6 : (tensor<1xi64>) -> tensor<1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<8x64x128xf32>
    %5 = stablehlo.power %1, %4 : tensor<8x64x128xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x64x128xf32>, tensor<f32>) -> tensor<8x64xf32>
    %7 = stablehlo.reshape %6 : (tensor<8x64xf32>) -> tensor<8x64x1xf32>
    %8 = stablehlo.convert %cst_7 : (tensor<1xi64>) -> tensor<1xf32>
    %9 = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<8x64x1xf32>
    %11 = stablehlo.divide %7, %10 : tensor<8x64x1xf32>
    %12 = stablehlo.convert %cst_8 : (tensor<1xf64>) -> tensor<1xf32>
    %13 = stablehlo.reshape %12 : (tensor<1xf32>) -> tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<8x64x1xf32>
    %15 = stablehlo.add %11, %14 : tensor<8x64x1xf32>
    %16 = stablehlo.rsqrt %15 : tensor<8x64x1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<8x64x1xf32>) -> tensor<8x64x128xf32>
    %18 = stablehlo.multiply %1, %17 : tensor<8x64x128xf32>
    %19 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<128xf32>) -> tensor<8x64x128xf32>
    %20 = stablehlo.multiply %18, %19 : tensor<8x64x128xf32>
    %21 = stablehlo.transpose %arg6, dims = [1, 0] : (tensor<384x128xf32>) -> tensor<128x384xf32>
    %22 = stablehlo.reshape %20 : (tensor<8x64x128xf32>) -> tensor<512x128xf32>
    %23 = stablehlo.dot_general %22, %21, contracting_dims = [1] x [0] : (tensor<512x128xf32>, tensor<128x384xf32>) -> tensor<512x384xf32>
    %24 = stablehlo.reshape %23 : (tensor<512x384xf32>) -> tensor<8x64x384xf32>
    %25 = stablehlo.slice %24 [0:8, 0:64, 0:128] : (tensor<8x64x384xf32>) -> tensor<8x64x128xf32>
    %26 = stablehlo.slice %24 [0:8, 0:64, 128:256] : (tensor<8x64x384xf32>) -> tensor<8x64x128xf32>
    %27 = stablehlo.slice %24 [0:8, 0:64, 256:384] : (tensor<8x64x384xf32>) -> tensor<8x64x128xf32>
    %28 = stablehlo.reshape %25 : (tensor<8x64x128xf32>) -> tensor<8x64x4x32xf32>
    %29 = stablehlo.reshape %26 : (tensor<8x64x128xf32>) -> tensor<8x64x4x32xf32>
    %30 = stablehlo.reshape %27 : (tensor<8x64x128xf32>) -> tensor<8x64x4x32xf32>
    %31 = stablehlo.transpose %30, dims = [0, 2, 1, 3] : (tensor<8x64x4x32xf32>) -> tensor<8x4x64x32xf32>
    %32 = stablehlo.slice %28 [0:8, 0:64, 0:4, 0:16] : (tensor<8x64x4x32xf32>) -> tensor<8x64x4x16xf32>
    %33 = stablehlo.slice %28 [0:8, 0:64, 0:4, 16:32] : (tensor<8x64x4x32xf32>) -> tensor<8x64x4x16xf32>
    %34 = stablehlo.reshape %arg20 : (tensor<64x16xf32>) -> tensor<1x64x1x16xf32>
    %35 = stablehlo.reshape %arg21 : (tensor<64x16xf32>) -> tensor<1x64x1x16xf32>
    %36 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2, 3] : (tensor<1x64x1x16xf32>) -> tensor<8x64x4x16xf32>
    %37 = stablehlo.multiply %32, %36 : tensor<8x64x4x16xf32>
    %38 = stablehlo.broadcast_in_dim %35, dims = [0, 1, 2, 3] : (tensor<1x64x1x16xf32>) -> tensor<8x64x4x16xf32>
    %39 = stablehlo.multiply %33, %38 : tensor<8x64x4x16xf32>
    %40 = stablehlo.subtract %37, %39 : tensor<8x64x4x16xf32>
    %41 = stablehlo.multiply %33, %36 : tensor<8x64x4x16xf32>
    %42 = stablehlo.multiply %32, %38 : tensor<8x64x4x16xf32>
    %43 = stablehlo.add %41, %42 : tensor<8x64x4x16xf32>
    %44 = stablehlo.concatenate %40, %43, dim = 3 : (tensor<8x64x4x16xf32>, tensor<8x64x4x16xf32>) -> tensor<8x64x4x32xf32>
    %45 = stablehlo.transpose %44, dims = [0, 2, 1, 3] : (tensor<8x64x4x32xf32>) -> tensor<8x4x64x32xf32>
    %46 = stablehlo.slice %29 [0:8, 0:64, 0:4, 0:16] : (tensor<8x64x4x32xf32>) -> tensor<8x64x4x16xf32>
    %47 = stablehlo.slice %29 [0:8, 0:64, 0:4, 16:32] : (tensor<8x64x4x32xf32>) -> tensor<8x64x4x16xf32>
    %48 = stablehlo.multiply %46, %36 : tensor<8x64x4x16xf32>
    %49 = stablehlo.multiply %47, %38 : tensor<8x64x4x16xf32>
    %50 = stablehlo.subtract %48, %49 : tensor<8x64x4x16xf32>
    %51 = stablehlo.multiply %47, %36 : tensor<8x64x4x16xf32>
    %52 = stablehlo.multiply %46, %38 : tensor<8x64x4x16xf32>
    %53 = stablehlo.add %51, %52 : tensor<8x64x4x16xf32>
    %54 = stablehlo.concatenate %50, %53, dim = 3 : (tensor<8x64x4x16xf32>, tensor<8x64x4x16xf32>) -> tensor<8x64x4x32xf32>
    %55 = stablehlo.transpose %54, dims = [0, 2, 1, 3] : (tensor<8x64x4x32xf32>) -> tensor<8x4x64x32xf32>
    %56 = stablehlo.transpose %55, dims = [0, 1, 3, 2] : (tensor<8x4x64x32xf32>) -> tensor<8x4x32x64xf32>
    %57 = stablehlo.reshape %45 : (tensor<8x4x64x32xf32>) -> tensor<32x64x32xf32>
    %58 = stablehlo.reshape %56 : (tensor<8x4x32x64xf32>) -> tensor<32x32x64xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2] : (tensor<32x32x64xf32>) -> tensor<32x32x64xf32>
    %60 = stablehlo.dot_general %57, %59, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<32x64x32xf32>, tensor<32x32x64xf32>) -> tensor<32x64x64xf32>
    %61 = stablehlo.reshape %60 : (tensor<32x64x64xf32>) -> tensor<8x4x64x64xf32>
    %62 = stablehlo.convert %cst_9 : (tensor<1xf64>) -> tensor<1xf32>
    %63 = stablehlo.reshape %62 : (tensor<1xf32>) -> tensor<f32>
    %64 = stablehlo.broadcast_in_dim %63, dims = [] : (tensor<f32>) -> tensor<8x4x64x64xf32>
    %65 = stablehlo.multiply %61, %64 : tensor<8x4x64x64xf32>
    %66 = stablehlo.iota dim = 1 : tensor<64x64xi64>
    %67 = stablehlo.iota dim = 0 : tensor<64x64xi64>
    %68 = stablehlo.add %67, %c_1 : tensor<64x64xi64>
    %69 = stablehlo.compare  LE, %66, %68,  SIGNED : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %70 = stablehlo.broadcast_in_dim %69, dims = [0, 1] : (tensor<64x64xi1>) -> tensor<64x64xi1>
    %71 = stablehlo.select %70, %cst_4, %cst_5 : tensor<64x64xi1>, tensor<64x64xf32>
    %72 = stablehlo.reshape %71 : (tensor<64x64xf32>) -> tensor<1x1x64x64xf32>
    %73 = stablehlo.convert %c_2 : (tensor<i64>) -> tensor<f32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [] : (tensor<f32>) -> tensor<1x1x64x64xf32>
    %75 = stablehlo.compare  EQ, %72, %74,  FLOAT : (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>) -> tensor<1x1x64x64xi1>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x1x64x64xi1>) -> tensor<8x4x64x64xi1>
    %77 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x4x64x64xf32>
    %78 = stablehlo.broadcast_in_dim %65, dims = [0, 1, 2, 3] : (tensor<8x4x64x64xf32>) -> tensor<8x4x64x64xf32>
    %79 = stablehlo.select %76, %77, %78 : tensor<8x4x64x64xi1>, tensor<8x4x64x64xf32>
    %80 = stablehlo.reduce(%79 init: %cst_3) applies stablehlo.maximum across dimensions = [3] : (tensor<8x4x64x64xf32>, tensor<f32>) -> tensor<8x4x64xf32>
    %81 = stablehlo.reshape %80 : (tensor<8x4x64xf32>) -> tensor<8x4x64x1xf32>
    %82 = stablehlo.broadcast_in_dim %81, dims = [0, 1, 2, 3] : (tensor<8x4x64x1xf32>) -> tensor<8x4x64x64xf32>
    %83 = stablehlo.subtract %79, %82 : tensor<8x4x64x64xf32>
    %84 = stablehlo.exponential %83 : tensor<8x4x64x64xf32>
    %85 = stablehlo.reduce(%84 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x4x64x64xf32>, tensor<f32>) -> tensor<8x4x64xf32>
    %86 = stablehlo.reshape %85 : (tensor<8x4x64xf32>) -> tensor<8x4x64x1xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1, 2, 3] : (tensor<8x4x64x1xf32>) -> tensor<8x4x64x64xf32>
    %88 = stablehlo.divide %84, %87 : tensor<8x4x64x64xf32>
    %89 = stablehlo.reshape %88 : (tensor<8x4x64x64xf32>) -> tensor<32x64x64xf32>
    %90 = stablehlo.reshape %31 : (tensor<8x4x64x32xf32>) -> tensor<32x64x32xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<32x64x32xf32>) -> tensor<32x64x32xf32>
    %92 = stablehlo.dot_general %89, %91, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<32x64x64xf32>, tensor<32x64x32xf32>) -> tensor<32x64x32xf32>
    %93 = stablehlo.reshape %92 : (tensor<32x64x32xf32>) -> tensor<8x4x64x32xf32>
    %94 = stablehlo.transpose %93, dims = [0, 2, 1, 3] : (tensor<8x4x64x32xf32>) -> tensor<8x64x4x32xf32>
    %95 = stablehlo.reshape %94 : (tensor<8x64x4x32xf32>) -> tensor<8x64x128xf32>
    %96 = stablehlo.transpose %arg7, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %97 = stablehlo.reshape %95 : (tensor<8x64x128xf32>) -> tensor<512x128xf32>
    %98 = stablehlo.dot_general %97, %96, contracting_dims = [1] x [0] : (tensor<512x128xf32>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %99 = stablehlo.reshape %98 : (tensor<512x128xf32>) -> tensor<8x64x128xf32>
    %100 = stablehlo.add %1, %99 : tensor<8x64x128xf32>
    %101 = stablehlo.power %100, %4 : tensor<8x64x128xf32>
    %102 = stablehlo.reduce(%101 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x64x128xf32>, tensor<f32>) -> tensor<8x64xf32>
    %103 = stablehlo.reshape %102 : (tensor<8x64xf32>) -> tensor<8x64x1xf32>
    %104 = stablehlo.divide %103, %10 : tensor<8x64x1xf32>
    %105 = stablehlo.add %104, %14 : tensor<8x64x1xf32>
    %106 = stablehlo.rsqrt %105 : tensor<8x64x1xf32>
    %107 = stablehlo.broadcast_in_dim %106, dims = [0, 1, 2] : (tensor<8x64x1xf32>) -> tensor<8x64x128xf32>
    %108 = stablehlo.multiply %100, %107 : tensor<8x64x128xf32>
    %109 = stablehlo.broadcast_in_dim %arg8, dims = [2] : (tensor<128xf32>) -> tensor<8x64x128xf32>
    %110 = stablehlo.multiply %108, %109 : tensor<8x64x128xf32>
    %111 = stablehlo.transpose %arg9, dims = [1, 0] : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %112 = stablehlo.reshape %110 : (tensor<8x64x128xf32>) -> tensor<512x128xf32>
    %113 = stablehlo.dot_general %112, %111, contracting_dims = [1] x [0] : (tensor<512x128xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %114 = stablehlo.reshape %113 : (tensor<512x512xf32>) -> tensor<8x64x512xf32>
    %115 = stablehlo.logistic %114 : tensor<8x64x512xf32>
    %116 = stablehlo.multiply %115, %114 : tensor<8x64x512xf32>
    %117 = stablehlo.transpose %arg10, dims = [1, 0] : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %118 = stablehlo.dot_general %112, %117, contracting_dims = [1] x [0] : (tensor<512x128xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %119 = stablehlo.reshape %118 : (tensor<512x512xf32>) -> tensor<8x64x512xf32>
    %120 = stablehlo.multiply %116, %119 : tensor<8x64x512xf32>
    %121 = stablehlo.transpose %arg11, dims = [1, 0] : (tensor<128x512xf32>) -> tensor<512x128xf32>
    %122 = stablehlo.reshape %120 : (tensor<8x64x512xf32>) -> tensor<512x512xf32>
    %123 = stablehlo.dot_general %122, %121, contracting_dims = [1] x [0] : (tensor<512x512xf32>, tensor<512x128xf32>) -> tensor<512x128xf32>
    %124 = stablehlo.reshape %123 : (tensor<512x128xf32>) -> tensor<8x64x128xf32>
    %125 = stablehlo.add %100, %124 : tensor<8x64x128xf32>
    %126 = stablehlo.power %125, %4 : tensor<8x64x128xf32>
    %127 = stablehlo.reduce(%126 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x64x128xf32>, tensor<f32>) -> tensor<8x64xf32>
    %128 = stablehlo.reshape %127 : (tensor<8x64xf32>) -> tensor<8x64x1xf32>
    %129 = stablehlo.divide %128, %10 : tensor<8x64x1xf32>
    %130 = stablehlo.add %129, %14 : tensor<8x64x1xf32>
    %131 = stablehlo.rsqrt %130 : tensor<8x64x1xf32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [0, 1, 2] : (tensor<8x64x1xf32>) -> tensor<8x64x128xf32>
    %133 = stablehlo.multiply %125, %132 : tensor<8x64x128xf32>
    %134 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<128xf32>) -> tensor<8x64x128xf32>
    %135 = stablehlo.multiply %133, %134 : tensor<8x64x128xf32>
    %136 = stablehlo.transpose %arg13, dims = [1, 0] : (tensor<384x128xf32>) -> tensor<128x384xf32>
    %137 = stablehlo.reshape %135 : (tensor<8x64x128xf32>) -> tensor<512x128xf32>
    %138 = stablehlo.dot_general %137, %136, contracting_dims = [1] x [0] : (tensor<512x128xf32>, tensor<128x384xf32>) -> tensor<512x384xf32>
    %139 = stablehlo.reshape %138 : (tensor<512x384xf32>) -> tensor<8x64x384xf32>
    %140 = stablehlo.slice %139 [0:8, 0:64, 0:128] : (tensor<8x64x384xf32>) -> tensor<8x64x128xf32>
    %141 = stablehlo.slice %139 [0:8, 0:64, 128:256] : (tensor<8x64x384xf32>) -> tensor<8x64x128xf32>
    %142 = stablehlo.slice %139 [0:8, 0:64, 256:384] : (tensor<8x64x384xf32>) -> tensor<8x64x128xf32>
    %143 = stablehlo.reshape %140 : (tensor<8x64x128xf32>) -> tensor<8x64x4x32xf32>
    %144 = stablehlo.reshape %141 : (tensor<8x64x128xf32>) -> tensor<8x64x4x32xf32>
    %145 = stablehlo.reshape %142 : (tensor<8x64x128xf32>) -> tensor<8x64x4x32xf32>
    %146 = stablehlo.transpose %145, dims = [0, 2, 1, 3] : (tensor<8x64x4x32xf32>) -> tensor<8x4x64x32xf32>
    %147 = stablehlo.slice %143 [0:8, 0:64, 0:4, 0:16] : (tensor<8x64x4x32xf32>) -> tensor<8x64x4x16xf32>
    %148 = stablehlo.slice %143 [0:8, 0:64, 0:4, 16:32] : (tensor<8x64x4x32xf32>) -> tensor<8x64x4x16xf32>
    %149 = stablehlo.reshape %arg2 : (tensor<64x16xf32>) -> tensor<1x64x1x16xf32>
    %150 = stablehlo.reshape %arg3 : (tensor<64x16xf32>) -> tensor<1x64x1x16xf32>
    %151 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2, 3] : (tensor<1x64x1x16xf32>) -> tensor<8x64x4x16xf32>
    %152 = stablehlo.multiply %147, %151 : tensor<8x64x4x16xf32>
    %153 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2, 3] : (tensor<1x64x1x16xf32>) -> tensor<8x64x4x16xf32>
    %154 = stablehlo.multiply %148, %153 : tensor<8x64x4x16xf32>
    %155 = stablehlo.subtract %152, %154 : tensor<8x64x4x16xf32>
    %156 = stablehlo.multiply %148, %151 : tensor<8x64x4x16xf32>
    %157 = stablehlo.multiply %147, %153 : tensor<8x64x4x16xf32>
    %158 = stablehlo.add %156, %157 : tensor<8x64x4x16xf32>
    %159 = stablehlo.concatenate %155, %158, dim = 3 : (tensor<8x64x4x16xf32>, tensor<8x64x4x16xf32>) -> tensor<8x64x4x32xf32>
    %160 = stablehlo.transpose %159, dims = [0, 2, 1, 3] : (tensor<8x64x4x32xf32>) -> tensor<8x4x64x32xf32>
    %161 = stablehlo.slice %144 [0:8, 0:64, 0:4, 0:16] : (tensor<8x64x4x32xf32>) -> tensor<8x64x4x16xf32>
    %162 = stablehlo.slice %144 [0:8, 0:64, 0:4, 16:32] : (tensor<8x64x4x32xf32>) -> tensor<8x64x4x16xf32>
    %163 = stablehlo.multiply %161, %151 : tensor<8x64x4x16xf32>
    %164 = stablehlo.multiply %162, %153 : tensor<8x64x4x16xf32>
    %165 = stablehlo.subtract %163, %164 : tensor<8x64x4x16xf32>
    %166 = stablehlo.multiply %162, %151 : tensor<8x64x4x16xf32>
    %167 = stablehlo.multiply %161, %153 : tensor<8x64x4x16xf32>
    %168 = stablehlo.add %166, %167 : tensor<8x64x4x16xf32>
    %169 = stablehlo.concatenate %165, %168, dim = 3 : (tensor<8x64x4x16xf32>, tensor<8x64x4x16xf32>) -> tensor<8x64x4x32xf32>
    %170 = stablehlo.transpose %169, dims = [0, 2, 1, 3] : (tensor<8x64x4x32xf32>) -> tensor<8x4x64x32xf32>
    %171 = stablehlo.transpose %170, dims = [0, 1, 3, 2] : (tensor<8x4x64x32xf32>) -> tensor<8x4x32x64xf32>
    %172 = stablehlo.reshape %160 : (tensor<8x4x64x32xf32>) -> tensor<32x64x32xf32>
    %173 = stablehlo.reshape %171 : (tensor<8x4x32x64xf32>) -> tensor<32x32x64xf32>
    %174 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2] : (tensor<32x32x64xf32>) -> tensor<32x32x64xf32>
    %175 = stablehlo.dot_general %172, %174, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<32x64x32xf32>, tensor<32x32x64xf32>) -> tensor<32x64x64xf32>
    %176 = stablehlo.reshape %175 : (tensor<32x64x64xf32>) -> tensor<8x4x64x64xf32>
    %177 = stablehlo.multiply %176, %64 : tensor<8x4x64x64xf32>
    %178 = stablehlo.broadcast_in_dim %177, dims = [0, 1, 2, 3] : (tensor<8x4x64x64xf32>) -> tensor<8x4x64x64xf32>
    %179 = stablehlo.select %76, %77, %178 : tensor<8x4x64x64xi1>, tensor<8x4x64x64xf32>
    %180 = stablehlo.reduce(%179 init: %cst_3) applies stablehlo.maximum across dimensions = [3] : (tensor<8x4x64x64xf32>, tensor<f32>) -> tensor<8x4x64xf32>
    %181 = stablehlo.reshape %180 : (tensor<8x4x64xf32>) -> tensor<8x4x64x1xf32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2, 3] : (tensor<8x4x64x1xf32>) -> tensor<8x4x64x64xf32>
    %183 = stablehlo.subtract %179, %182 : tensor<8x4x64x64xf32>
    %184 = stablehlo.exponential %183 : tensor<8x4x64x64xf32>
    %185 = stablehlo.reduce(%184 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<8x4x64x64xf32>, tensor<f32>) -> tensor<8x4x64xf32>
    %186 = stablehlo.reshape %185 : (tensor<8x4x64xf32>) -> tensor<8x4x64x1xf32>
    %187 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2, 3] : (tensor<8x4x64x1xf32>) -> tensor<8x4x64x64xf32>
    %188 = stablehlo.divide %184, %187 : tensor<8x4x64x64xf32>
    %189 = stablehlo.reshape %188 : (tensor<8x4x64x64xf32>) -> tensor<32x64x64xf32>
    %190 = stablehlo.reshape %146 : (tensor<8x4x64x32xf32>) -> tensor<32x64x32xf32>
    %191 = stablehlo.broadcast_in_dim %190, dims = [0, 1, 2] : (tensor<32x64x32xf32>) -> tensor<32x64x32xf32>
    %192 = stablehlo.dot_general %189, %191, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<32x64x64xf32>, tensor<32x64x32xf32>) -> tensor<32x64x32xf32>
    %193 = stablehlo.reshape %192 : (tensor<32x64x32xf32>) -> tensor<8x4x64x32xf32>
    %194 = stablehlo.transpose %193, dims = [0, 2, 1, 3] : (tensor<8x4x64x32xf32>) -> tensor<8x64x4x32xf32>
    %195 = stablehlo.reshape %194 : (tensor<8x64x4x32xf32>) -> tensor<8x64x128xf32>
    %196 = stablehlo.transpose %arg14, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %197 = stablehlo.reshape %195 : (tensor<8x64x128xf32>) -> tensor<512x128xf32>
    %198 = stablehlo.dot_general %197, %196, contracting_dims = [1] x [0] : (tensor<512x128xf32>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %199 = stablehlo.reshape %198 : (tensor<512x128xf32>) -> tensor<8x64x128xf32>
    %200 = stablehlo.add %125, %199 : tensor<8x64x128xf32>
    %201 = stablehlo.power %200, %4 : tensor<8x64x128xf32>
    %202 = stablehlo.reduce(%201 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x64x128xf32>, tensor<f32>) -> tensor<8x64xf32>
    %203 = stablehlo.reshape %202 : (tensor<8x64xf32>) -> tensor<8x64x1xf32>
    %204 = stablehlo.divide %203, %10 : tensor<8x64x1xf32>
    %205 = stablehlo.add %204, %14 : tensor<8x64x1xf32>
    %206 = stablehlo.rsqrt %205 : tensor<8x64x1xf32>
    %207 = stablehlo.broadcast_in_dim %206, dims = [0, 1, 2] : (tensor<8x64x1xf32>) -> tensor<8x64x128xf32>
    %208 = stablehlo.multiply %200, %207 : tensor<8x64x128xf32>
    %209 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<128xf32>) -> tensor<8x64x128xf32>
    %210 = stablehlo.multiply %208, %209 : tensor<8x64x128xf32>
    %211 = stablehlo.transpose %arg16, dims = [1, 0] : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %212 = stablehlo.reshape %210 : (tensor<8x64x128xf32>) -> tensor<512x128xf32>
    %213 = stablehlo.dot_general %212, %211, contracting_dims = [1] x [0] : (tensor<512x128xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %214 = stablehlo.reshape %213 : (tensor<512x512xf32>) -> tensor<8x64x512xf32>
    %215 = stablehlo.logistic %214 : tensor<8x64x512xf32>
    %216 = stablehlo.multiply %215, %214 : tensor<8x64x512xf32>
    %217 = stablehlo.transpose %arg17, dims = [1, 0] : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %218 = stablehlo.dot_general %212, %217, contracting_dims = [1] x [0] : (tensor<512x128xf32>, tensor<128x512xf32>) -> tensor<512x512xf32>
    %219 = stablehlo.reshape %218 : (tensor<512x512xf32>) -> tensor<8x64x512xf32>
    %220 = stablehlo.multiply %216, %219 : tensor<8x64x512xf32>
    %221 = stablehlo.transpose %arg18, dims = [1, 0] : (tensor<128x512xf32>) -> tensor<512x128xf32>
    %222 = stablehlo.reshape %220 : (tensor<8x64x512xf32>) -> tensor<512x512xf32>
    %223 = stablehlo.dot_general %222, %221, contracting_dims = [1] x [0] : (tensor<512x512xf32>, tensor<512x128xf32>) -> tensor<512x128xf32>
    %224 = stablehlo.reshape %223 : (tensor<512x128xf32>) -> tensor<8x64x128xf32>
    %225 = stablehlo.add %200, %224 : tensor<8x64x128xf32>
    %226 = stablehlo.power %225, %4 : tensor<8x64x128xf32>
    %227 = stablehlo.reduce(%226 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<8x64x128xf32>, tensor<f32>) -> tensor<8x64xf32>
    %228 = stablehlo.reshape %227 : (tensor<8x64xf32>) -> tensor<8x64x1xf32>
    %229 = stablehlo.divide %228, %10 : tensor<8x64x1xf32>
    %230 = stablehlo.add %229, %14 : tensor<8x64x1xf32>
    %231 = stablehlo.rsqrt %230 : tensor<8x64x1xf32>
    %232 = stablehlo.broadcast_in_dim %231, dims = [0, 1, 2] : (tensor<8x64x1xf32>) -> tensor<8x64x128xf32>
    %233 = stablehlo.multiply %225, %232 : tensor<8x64x128xf32>
    %234 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<128xf32>) -> tensor<8x64x128xf32>
    %235 = stablehlo.multiply %233, %234 : tensor<8x64x128xf32>
    %236 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<256x128xf32>) -> tensor<128x256xf32>
    %237 = stablehlo.reshape %235 : (tensor<8x64x128xf32>) -> tensor<512x128xf32>
    %238 = stablehlo.dot_general %237, %236, contracting_dims = [1] x [0] : (tensor<512x128xf32>, tensor<128x256xf32>) -> tensor<512x256xf32>
    %239 = stablehlo.reshape %238 : (tensor<512x256xf32>) -> tensor<8x64x256xf32>
    %240 = stablehlo.reshape %239 : (tensor<8x64x256xf32>) -> tensor<512x256xf32>
    %241 = stablehlo.reshape %arg23 : (tensor<8x64xi64>) -> tensor<512xi64>
    %242 = stablehlo.reduce(%240 init: %cst_3) applies stablehlo.maximum across dimensions = [1] : (tensor<512x256xf32>, tensor<f32>) -> tensor<512xf32>
    %243 = stablehlo.reshape %242 : (tensor<512xf32>) -> tensor<512x1xf32>
    %244 = stablehlo.broadcast_in_dim %243, dims = [0, 1] : (tensor<512x1xf32>) -> tensor<512x256xf32>
    %245 = stablehlo.subtract %240, %244 : tensor<512x256xf32>
    %246 = stablehlo.exponential %245 : tensor<512x256xf32>
    %247 = stablehlo.reduce(%246 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<512x256xf32>, tensor<f32>) -> tensor<512xf32>
    %248 = stablehlo.reshape %247 : (tensor<512xf32>) -> tensor<512x1xf32>
    %249 = stablehlo.log %248 : tensor<512x1xf32>
    %250 = stablehlo.broadcast_in_dim %249, dims = [0, 1] : (tensor<512x1xf32>) -> tensor<512x256xf32>
    %251 = stablehlo.subtract %245, %250 : tensor<512x256xf32>
    %252 = stablehlo.compare  NE, %241, %c_0,  SIGNED : (tensor<512xi64>, tensor<512xi64>) -> tensor<512xi1>
    %253 = stablehlo.broadcast_in_dim %252, dims = [0] : (tensor<512xi1>) -> tensor<512xi1>
    %254 = stablehlo.broadcast_in_dim %241, dims = [0] : (tensor<512xi64>) -> tensor<512xi64>
    %255 = stablehlo.select %253, %254, %c : tensor<512xi1>, tensor<512xi64>
    %256 = stablehlo.reshape %255 : (tensor<512xi64>) -> tensor<512x1xi64>
    %257 = stablehlo.iota dim = 0 : tensor<512x1x1xi64>
    %258 = stablehlo.reshape %256 : (tensor<512x1xi64>) -> tensor<512x1x1xi64>
    %259 = stablehlo.concatenate %257, %258, dim = 2 : (tensor<512x1x1xi64>, tensor<512x1x1xi64>) -> tensor<512x1x2xi64>
    %260 = "stablehlo.gather"(%251, %259) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<512x256xf32>, tensor<512x1x2xi64>) -> tensor<512x1xf32>
    %261 = stablehlo.reshape %260 : (tensor<512x1xf32>) -> tensor<512xf32>
    %262 = stablehlo.negate %261 : tensor<512xf32>
    %263 = stablehlo.broadcast_in_dim %262, dims = [0] : (tensor<512xf32>) -> tensor<512xf32>
    %264 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %265 = stablehlo.select %253, %263, %264 : tensor<512xi1>, tensor<512xf32>
    %266 = stablehlo.convert %252 : (tensor<512xi1>) -> tensor<512xi64>
    %267 = stablehlo.reduce(%266 init: %c_2) applies stablehlo.add across dimensions = [0] : (tensor<512xi64>, tensor<i64>) -> tensor<i64>
    %268 = stablehlo.convert %267 : (tensor<i64>) -> tensor<f32>
    %269 = stablehlo.reduce(%265 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<512xf32>, tensor<f32>) -> tensor<f32>
    %270 = stablehlo.divide %269, %268 : tensor<f32>
    return %270 : tensor<f32>
  }
}
