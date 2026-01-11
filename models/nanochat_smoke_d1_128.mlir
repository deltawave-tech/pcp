module {
  func.func @main(%arg0: tensor<65536x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>, %arg5: tensor<256x64xf32>, %arg6: tensor<64x256xf32>, %arg7: tensor<65536x64xf32>, %arg8: tensor<1x128xi64>, %arg9: tensor<1x128xi64>) -> tensor<f32> {
    %c = stablehlo.constant dense<-1> : tensor<128xi64>
    %c_0 = stablehlo.constant dense<1> : tensor<128x128xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<128xi64>
    %c_2 = stablehlo.constant dense<1> : tensor<128xi64>
    %c_3 = stablehlo.constant dense<128> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c_5 = stablehlo.constant dense<0> : tensor<i64>
    %cst_6 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c_7 = stablehlo.constant dense<true> : tensor<128x128xi1>
    %cst_8 = stablehlo.constant dense_resource<torch_tensor_1_128_1_8_torch.bfloat16_1> : tensor<1x128x1x8xbf16>
    %cst_9 = stablehlo.constant dense_resource<torch_tensor_1_128_1_8_torch.bfloat16> : tensor<1x128x1x8xbf16>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<1x128x256xf32>
    %cst_11 = arith.constant dense<2> : tensor<1xi64>
    %cst_12 = arith.constant dense<64> : tensor<1xi64>
    %cst_13 = arith.constant dense<1.000000e-05> : tensor<1xf64>
    %cst_14 = arith.constant dense<16> : tensor<1xi64>
    %cst_15 = arith.constant dense<2.500000e-01> : tensor<1xf64>
    %cst_16 = arith.constant dense<15> : tensor<1xi64>
    %0 = "stablehlo.gather"(%arg0, %arg8) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 64>}> : (tensor<65536x64xf32>, tensor<1x128xi64>) -> tensor<1x128x64xf32>
    %1 = stablehlo.convert %0 : tensor<1x128x64xf32>
    %2 = stablehlo.convert %cst_11 : (tensor<1xi64>) -> tensor<1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1xf32>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x128x64xf32>
    %5 = stablehlo.power %1, %4 : tensor<1x128x64xf32>
    %6 = stablehlo.reduce(%5 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x128x64xf32>, tensor<f32>) -> tensor<1x128xf32>
    %7 = stablehlo.reshape %6 : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %8 = stablehlo.convert %cst_12 : (tensor<1xi64>) -> tensor<1xf32>
    %9 = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %11 = stablehlo.divide %7, %10 : tensor<1x128x1xf32>
    %12 = stablehlo.convert %cst_13 : (tensor<1xf64>) -> tensor<1xf32>
    %13 = stablehlo.reshape %12 : (tensor<1xf32>) -> tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %15 = stablehlo.add %11, %14 : tensor<1x128x1xf32>
    %16 = stablehlo.rsqrt %15 : tensor<1x128x1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x64xf32>
    %18 = stablehlo.multiply %1, %17 : tensor<1x128x64xf32>
    %19 = stablehlo.power %18, %4 : tensor<1x128x64xf32>
    %20 = stablehlo.reduce(%19 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x128x64xf32>, tensor<f32>) -> tensor<1x128xf32>
    %21 = stablehlo.reshape %20 : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %22 = stablehlo.divide %21, %10 : tensor<1x128x1xf32>
    %23 = stablehlo.add %22, %14 : tensor<1x128x1xf32>
    %24 = stablehlo.rsqrt %23 : tensor<1x128x1xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x64xf32>
    %26 = stablehlo.multiply %18, %25 : tensor<1x128x64xf32>
    %27 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %28 = stablehlo.reshape %26 : (tensor<1x128x64xf32>) -> tensor<128x64xf32>
    %29 = stablehlo.dot_general %28, %27, contracting_dims = [1] x [0] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %30 = stablehlo.reshape %29 : (tensor<128x64xf32>) -> tensor<1x128x64xf32>
    %31 = stablehlo.reshape %30 : (tensor<1x128x64xf32>) -> tensor<1x128x4x16xf32>
    %32 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %33 = stablehlo.dot_general %28, %32, contracting_dims = [1] x [0] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %34 = stablehlo.reshape %33 : (tensor<128x64xf32>) -> tensor<1x128x64xf32>
    %35 = stablehlo.reshape %34 : (tensor<1x128x64xf32>) -> tensor<1x128x4x16xf32>
    %36 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %37 = stablehlo.dot_general %28, %36, contracting_dims = [1] x [0] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %38 = stablehlo.reshape %37 : (tensor<128x64xf32>) -> tensor<1x128x64xf32>
    %39 = stablehlo.reshape %38 : (tensor<1x128x64xf32>) -> tensor<1x128x4x16xf32>
    %40 = stablehlo.slice %31 [0:1, 0:128, 0:4, 0:8] : (tensor<1x128x4x16xf32>) -> tensor<1x128x4x8xf32>
    %41 = stablehlo.slice %31 [0:1, 0:128, 0:4, 8:16] : (tensor<1x128x4x16xf32>) -> tensor<1x128x4x8xf32>
    %42 = stablehlo.convert %cst_9 : (tensor<1x128x1x8xbf16>) -> tensor<1x128x1x8xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1, 2, 3] : (tensor<1x128x1x8xf32>) -> tensor<1x128x4x8xf32>
    %44 = stablehlo.multiply %40, %43 : tensor<1x128x4x8xf32>
    %45 = stablehlo.convert %cst_8 : (tensor<1x128x1x8xbf16>) -> tensor<1x128x1x8xf32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x128x1x8xf32>) -> tensor<1x128x4x8xf32>
    %47 = stablehlo.multiply %41, %46 : tensor<1x128x4x8xf32>
    %48 = stablehlo.add %44, %47 : tensor<1x128x4x8xf32>
    %49 = stablehlo.negate %cst_8 : tensor<1x128x1x8xbf16>
    %50 = stablehlo.convert %49 : (tensor<1x128x1x8xbf16>) -> tensor<1x128x1x8xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2, 3] : (tensor<1x128x1x8xf32>) -> tensor<1x128x4x8xf32>
    %52 = stablehlo.multiply %40, %51 : tensor<1x128x4x8xf32>
    %53 = stablehlo.multiply %41, %43 : tensor<1x128x4x8xf32>
    %54 = stablehlo.add %52, %53 : tensor<1x128x4x8xf32>
    %55 = stablehlo.concatenate %48, %54, dim = 3 : (tensor<1x128x4x8xf32>, tensor<1x128x4x8xf32>) -> tensor<1x128x4x16xf32>
    %56 = stablehlo.slice %35 [0:1, 0:128, 0:4, 0:8] : (tensor<1x128x4x16xf32>) -> tensor<1x128x4x8xf32>
    %57 = stablehlo.slice %35 [0:1, 0:128, 0:4, 8:16] : (tensor<1x128x4x16xf32>) -> tensor<1x128x4x8xf32>
    %58 = stablehlo.multiply %56, %43 : tensor<1x128x4x8xf32>
    %59 = stablehlo.multiply %57, %46 : tensor<1x128x4x8xf32>
    %60 = stablehlo.add %58, %59 : tensor<1x128x4x8xf32>
    %61 = stablehlo.multiply %56, %51 : tensor<1x128x4x8xf32>
    %62 = stablehlo.multiply %57, %43 : tensor<1x128x4x8xf32>
    %63 = stablehlo.add %61, %62 : tensor<1x128x4x8xf32>
    %64 = stablehlo.concatenate %60, %63, dim = 3 : (tensor<1x128x4x8xf32>, tensor<1x128x4x8xf32>) -> tensor<1x128x4x16xf32>
    %65 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x128x4x16xf32>
    %66 = stablehlo.power %55, %65 : tensor<1x128x4x16xf32>
    %67 = stablehlo.reduce(%66 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x128x4x16xf32>, tensor<f32>) -> tensor<1x128x4xf32>
    %68 = stablehlo.reshape %67 : (tensor<1x128x4xf32>) -> tensor<1x128x4x1xf32>
    %69 = stablehlo.convert %cst_14 : (tensor<1xi64>) -> tensor<1xf32>
    %70 = stablehlo.reshape %69 : (tensor<1xf32>) -> tensor<f32>
    %71 = stablehlo.broadcast_in_dim %70, dims = [] : (tensor<f32>) -> tensor<1x128x4x1xf32>
    %72 = stablehlo.divide %68, %71 : tensor<1x128x4x1xf32>
    %73 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<1x128x4x1xf32>
    %74 = stablehlo.add %72, %73 : tensor<1x128x4x1xf32>
    %75 = stablehlo.rsqrt %74 : tensor<1x128x4x1xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x128x4x1xf32>) -> tensor<1x128x4x16xf32>
    %77 = stablehlo.multiply %55, %76 : tensor<1x128x4x16xf32>
    %78 = stablehlo.power %64, %65 : tensor<1x128x4x16xf32>
    %79 = stablehlo.reduce(%78 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x128x4x16xf32>, tensor<f32>) -> tensor<1x128x4xf32>
    %80 = stablehlo.reshape %79 : (tensor<1x128x4xf32>) -> tensor<1x128x4x1xf32>
    %81 = stablehlo.divide %80, %71 : tensor<1x128x4x1xf32>
    %82 = stablehlo.add %81, %73 : tensor<1x128x4x1xf32>
    %83 = stablehlo.rsqrt %82 : tensor<1x128x4x1xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2, 3] : (tensor<1x128x4x1xf32>) -> tensor<1x128x4x16xf32>
    %85 = stablehlo.multiply %64, %84 : tensor<1x128x4x16xf32>
    %86 = stablehlo.transpose %77, dims = [0, 2, 1, 3] : (tensor<1x128x4x16xf32>) -> tensor<1x4x128x16xf32>
    %87 = stablehlo.transpose %85, dims = [0, 2, 1, 3] : (tensor<1x128x4x16xf32>) -> tensor<1x4x128x16xf32>
    %88 = stablehlo.transpose %39, dims = [0, 2, 1, 3] : (tensor<1x128x4x16xf32>) -> tensor<1x4x128x16xf32>
    %89 = stablehlo.transpose %87, dims = [0, 1, 3, 2] : (tensor<1x4x128x16xf32>) -> tensor<1x4x16x128xf32>
    %90 = stablehlo.reshape %86 : (tensor<1x4x128x16xf32>) -> tensor<4x128x16xf32>
    %91 = stablehlo.reshape %89 : (tensor<1x4x16x128xf32>) -> tensor<4x16x128xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1, 2] : (tensor<4x16x128xf32>) -> tensor<4x16x128xf32>
    %93 = stablehlo.dot_general %90, %92, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x128x16xf32>, tensor<4x16x128xf32>) -> tensor<4x128x128xf32>
    %94 = stablehlo.reshape %93 : (tensor<4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %95 = stablehlo.convert %cst_15 : (tensor<1xf64>) -> tensor<1xf32>
    %96 = stablehlo.reshape %95 : (tensor<1xf32>) -> tensor<f32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %98 = stablehlo.multiply %94, %97 : tensor<1x4x128x128xf32>
    %99 = stablehlo.convert %c_4 : (tensor<i64>) -> tensor<f64>
    %100 = stablehlo.convert %c_3 : (tensor<i64>) -> tensor<f64>
    %101 = stablehlo.divide %100, %99 : tensor<f64>
    %102 = stablehlo.ceil %101 : tensor<f64>
    %103 = stablehlo.convert %102 : (tensor<f64>) -> tensor<i64>
    %104 = stablehlo.reshape %103 : (tensor<i64>) -> tensor<1xi64>
    %105 = stablehlo.dynamic_iota %104, dim = 0 : (tensor<1xi64>) -> tensor<128xi64>
    %106 = stablehlo.multiply %105, %c_2 : tensor<128xi64>
    %107 = stablehlo.add %106, %c_1 : tensor<128xi64>
    %108 = stablehlo.reshape %107 : (tensor<128xi64>) -> tensor<1x128xi64>
    %109 = stablehlo.reshape %107 : (tensor<128xi64>) -> tensor<128x1xi64>
    %110 = stablehlo.broadcast_in_dim %108, dims = [0, 1] : (tensor<1x128xi64>) -> tensor<128x128xi64>
    %111 = stablehlo.broadcast_in_dim %109, dims = [0, 1] : (tensor<128x1xi64>) -> tensor<128x128xi64>
    %112 = stablehlo.subtract %110, %111 : tensor<128x128xi64>
    %113 = stablehlo.compare  GE, %112, %c_0,  SIGNED : (tensor<128x128xi64>, tensor<128x128xi64>) -> tensor<128x128xi1>
    %114 = stablehlo.and %113, %c_7 : tensor<128x128xi1>
    %115 = stablehlo.reshape %114 : (tensor<128x128xi1>) -> tensor<1x1x128x128xi1>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0, 1, 2, 3] : (tensor<1x1x128x128xi1>) -> tensor<1x4x128x128xi1>
    %117 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %118 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2, 3] : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %119 = stablehlo.select %116, %117, %118 : tensor<1x4x128x128xi1>, tensor<1x4x128x128xf32>
    %120 = stablehlo.reduce(%119 init: %cst_6) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %121 = stablehlo.reshape %120 : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %122 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %123 = stablehlo.subtract %119, %122 : tensor<1x4x128x128xf32>
    %124 = stablehlo.exponential %123 : tensor<1x4x128x128xf32>
    %125 = stablehlo.reduce(%124 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %126 = stablehlo.reshape %125 : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %127 = stablehlo.broadcast_in_dim %126, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %128 = stablehlo.divide %124, %127 : tensor<1x4x128x128xf32>
    %129 = stablehlo.reshape %128 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %130 = stablehlo.reshape %88 : (tensor<1x4x128x16xf32>) -> tensor<4x128x16xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2] : (tensor<4x128x16xf32>) -> tensor<4x128x16xf32>
    %132 = stablehlo.dot_general %129, %131, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x128x128xf32>, tensor<4x128x16xf32>) -> tensor<4x128x16xf32>
    %133 = stablehlo.reshape %132 : (tensor<4x128x16xf32>) -> tensor<1x4x128x16xf32>
    %134 = stablehlo.transpose %133, dims = [0, 2, 1, 3] : (tensor<1x4x128x16xf32>) -> tensor<1x128x4x16xf32>
    %135 = stablehlo.reshape %134 : (tensor<1x128x4x16xf32>) -> tensor<1x128x64xf32>
    %136 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %137 = stablehlo.reshape %135 : (tensor<1x128x64xf32>) -> tensor<128x64xf32>
    %138 = stablehlo.dot_general %137, %136, contracting_dims = [1] x [0] : (tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<128x64xf32>
    %139 = stablehlo.reshape %138 : (tensor<128x64xf32>) -> tensor<1x128x64xf32>
    %140 = stablehlo.add %18, %139 : tensor<1x128x64xf32>
    %141 = stablehlo.power %140, %4 : tensor<1x128x64xf32>
    %142 = stablehlo.reduce(%141 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x128x64xf32>, tensor<f32>) -> tensor<1x128xf32>
    %143 = stablehlo.reshape %142 : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %144 = stablehlo.divide %143, %10 : tensor<1x128x1xf32>
    %145 = stablehlo.add %144, %14 : tensor<1x128x1xf32>
    %146 = stablehlo.rsqrt %145 : tensor<1x128x1xf32>
    %147 = stablehlo.broadcast_in_dim %146, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x64xf32>
    %148 = stablehlo.multiply %140, %147 : tensor<1x128x64xf32>
    %149 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<256x64xf32>) -> tensor<64x256xf32>
    %150 = stablehlo.reshape %148 : (tensor<1x128x64xf32>) -> tensor<128x64xf32>
    %151 = stablehlo.dot_general %150, %149, contracting_dims = [1] x [0] : (tensor<128x64xf32>, tensor<64x256xf32>) -> tensor<128x256xf32>
    %152 = stablehlo.reshape %151 : (tensor<128x256xf32>) -> tensor<1x128x256xf32>
    %153 = stablehlo.maximum %152, %cst_10 : tensor<1x128x256xf32>
    %154 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x128x256xf32>
    %155 = stablehlo.power %153, %154 : tensor<1x128x256xf32>
    %156 = stablehlo.transpose %arg6, dims = [1, 0] : (tensor<64x256xf32>) -> tensor<256x64xf32>
    %157 = stablehlo.reshape %155 : (tensor<1x128x256xf32>) -> tensor<128x256xf32>
    %158 = stablehlo.dot_general %157, %156, contracting_dims = [1] x [0] : (tensor<128x256xf32>, tensor<256x64xf32>) -> tensor<128x64xf32>
    %159 = stablehlo.reshape %158 : (tensor<128x64xf32>) -> tensor<1x128x64xf32>
    %160 = stablehlo.add %140, %159 : tensor<1x128x64xf32>
    %161 = stablehlo.power %160, %4 : tensor<1x128x64xf32>
    %162 = stablehlo.reduce(%161 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x128x64xf32>, tensor<f32>) -> tensor<1x128xf32>
    %163 = stablehlo.reshape %162 : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %164 = stablehlo.divide %163, %10 : tensor<1x128x1xf32>
    %165 = stablehlo.add %164, %14 : tensor<1x128x1xf32>
    %166 = stablehlo.rsqrt %165 : tensor<1x128x1xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x64xf32>
    %168 = stablehlo.multiply %160, %167 : tensor<1x128x64xf32>
    %169 = stablehlo.transpose %arg7, dims = [1, 0] : (tensor<65536x64xf32>) -> tensor<64x65536xf32>
    %170 = stablehlo.reshape %168 : (tensor<1x128x64xf32>) -> tensor<128x64xf32>
    %171 = stablehlo.dot_general %170, %169, contracting_dims = [1] x [0] : (tensor<128x64xf32>, tensor<64x65536xf32>) -> tensor<128x65536xf32>
    %172 = stablehlo.reshape %171 : (tensor<128x65536xf32>) -> tensor<1x128x65536xf32>
    %173 = stablehlo.convert %cst_16 : (tensor<1xi64>) -> tensor<1xf32>
    %174 = stablehlo.reshape %173 : (tensor<1xf32>) -> tensor<f32>
    %175 = stablehlo.broadcast_in_dim %174, dims = [] : (tensor<f32>) -> tensor<1x128x65536xf32>
    %176 = stablehlo.divide %172, %175 : tensor<1x128x65536xf32>
    %177 = stablehlo.tanh %176 : tensor<1x128x65536xf32>
    %178 = stablehlo.multiply %177, %175 : tensor<1x128x65536xf32>
    %179 = stablehlo.reshape %178 : (tensor<1x128x65536xf32>) -> tensor<128x65536xf32>
    %180 = stablehlo.reshape %arg9 : (tensor<1x128xi64>) -> tensor<128xi64>
    %181 = stablehlo.reduce(%179 init: %cst_6) applies stablehlo.maximum across dimensions = [1] : (tensor<128x65536xf32>, tensor<f32>) -> tensor<128xf32>
    %182 = stablehlo.reshape %181 : (tensor<128xf32>) -> tensor<128x1xf32>
    %183 = stablehlo.broadcast_in_dim %182, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x65536xf32>
    %184 = stablehlo.subtract %179, %183 : tensor<128x65536xf32>
    %185 = stablehlo.exponential %184 : tensor<128x65536xf32>
    %186 = stablehlo.reduce(%185 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<128x65536xf32>, tensor<f32>) -> tensor<128xf32>
    %187 = stablehlo.reshape %186 : (tensor<128xf32>) -> tensor<128x1xf32>
    %188 = stablehlo.log %187 : tensor<128x1xf32>
    %189 = stablehlo.broadcast_in_dim %188, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x65536xf32>
    %190 = stablehlo.subtract %184, %189 : tensor<128x65536xf32>
    %191 = stablehlo.compare  NE, %180, %c,  SIGNED : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %192 = stablehlo.broadcast_in_dim %191, dims = [0] : (tensor<128xi1>) -> tensor<128xi1>
    %193 = stablehlo.broadcast_in_dim %180, dims = [0] : (tensor<128xi64>) -> tensor<128xi64>
    %194 = stablehlo.select %192, %193, %c_1 : tensor<128xi1>, tensor<128xi64>
    %195 = stablehlo.reshape %194 : (tensor<128xi64>) -> tensor<128x1xi64>
    %196 = stablehlo.iota dim = 0 : tensor<128x1x1xi64>
    %197 = stablehlo.reshape %195 : (tensor<128x1xi64>) -> tensor<128x1x1xi64>
    %198 = stablehlo.concatenate %196, %197, dim = 2 : (tensor<128x1x1xi64>, tensor<128x1x1xi64>) -> tensor<128x1x2xi64>
    %199 = "stablehlo.gather"(%190, %198) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<128x65536xf32>, tensor<128x1x2xi64>) -> tensor<128x1xf32>
    %200 = stablehlo.reshape %199 : (tensor<128x1xf32>) -> tensor<128xf32>
    %201 = stablehlo.negate %200 : tensor<128xf32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [0] : (tensor<128xf32>) -> tensor<128xf32>
    %203 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %204 = stablehlo.select %192, %202, %203 : tensor<128xi1>, tensor<128xf32>
    %205 = stablehlo.convert %191 : (tensor<128xi1>) -> tensor<128xi64>
    %206 = stablehlo.reduce(%205 init: %c_5) applies stablehlo.add across dimensions = [0] : (tensor<128xi64>, tensor<i64>) -> tensor<i64>
    %207 = stablehlo.convert %206 : (tensor<i64>) -> tensor<f32>
    %208 = stablehlo.reduce(%204 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<128xf32>, tensor<f32>) -> tensor<f32>
    %209 = stablehlo.divide %208, %207 : tensor<f32>
    return %209 : tensor<f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_1_128_1_8_torch.bfloat16_1: "0x0200000000000000000000000000000000000000573F9F3ECC3D023D243C4F3B833AA639693F173F4B3E813DA43CCF3B033B263A113E503F973EC23DF63C1B3C453B793A42BF743FC73E013E243D4F3C833BA63A75BF803FF53E213E4D3D823CA43BCF3A8FBE723F113F413E763D9B3CC53BF93A283F4D3F253F613E8F3DB53CE53B113B7D3F133F383F803EA43DCF3C033C263BD33E953E493F903EB83DE93C133C3B3B0BBFA9BC573F9F3ECC3D023D243C4F3B80BFA9BE643FAF3EE13D0E3D343C643B09BF1CBF6F3FBE3EF53D1B3D453C793BD73E53BF773FCD3E053E283D553C873B7E3F76BF7C3FDB3E0F3E353D653C913B263F80BF7F3FEA3E193E423D763C9B3B93BE71BF803FF83E233E4F3D833CA63B76BF4ABF7E3F033F2D3E5C3D8B3CB03B40BF0FBF793F0A3F373E693D933CBB3B193E8BBE723F113F413E763D9C3CC53B6A3F293D693F173F4B3E813DA43CCF3B563FB33E5D3F1E3F553E883DAC3CDA3B11BC203F4F3F243F5F3E8E3DB43CE43B59BF563F3F3F2A3F693E953DBC3CEE3B68BF773F2D3F303F733E9B3DC53CF93B08BE803F193F363F7D3EA23DCD3C023C433F6F3F043F3C3F843EA83DD53C073C753F463FDB3E413F893EAF3DDD3C0C3C8B3E0A3FAC3E463F8D3EB53DE53C113C2ABF813E753E4B3F923EBC3DEE3C163C7DBF7EBD113E503F973EC23DF63C1B3CCFBEBDBE2A3D553F9C3EC83DFE3C213C0D3F24BF6FBD593FA13ECF3D033D263C803F59BF22BE5D3FA63ED53D073D2B3C073F78BF83BE613FAB3EDC3D0B3D303CDBBE7FBFB4BE653FB03EE23D0F3D353C7EBF6DBFE3BE683FB43EE93D133D3B3C25BF43BF08BF6C3FB93EEF3D183D403C983E06BF1DBF6F3FBE3EF63D1C3D453C773F6DBE30BF723FC33EFC3D203D4A3C3F3FA93D42BF743FC73E013E243D4F3C22BEC73E51BF763FCC3E043E283D543C6BBF283F5FBF793FD13E083E2C3D5A3C55BF5C3F6BBF7A3FD53E0B3E303D5F3C913C7A3F74BF7C3FDA3E0E3E343D643C5A3F7F3F7ABF7D3FDF3E113E383D693C673F6B3F7EBF7E3FE33E143E3C3D6E3CFD3D403F80BF7F3FE83E183E403D743C45BF013F7FBF803FEC3E1B3E453D793C74BF583E7CBF803FF13E1E3E493D7E3C86BED3BD75BF803FF53E213E4D3D823C2C3FD1BE6DBF803FFA3E243E513D843C7D3F2CBF62BF7F3FFE3E283E553D873CCB3E5EBF55BF7F3F013F2B3E593D893C0FBF7BBF46BF7E3F043F2E3E5D3D8C3C80BF7EBF35BF7C3F063F313E613D8E3C06BF69BF22BF7B3F083F343E653D913CDF3E3CBF0DBF793F0A3F383E693D943C7E3FF9BEEEBE773F0C3F3B3E6D3D963C233F44BEBFBE753F0E3F3E3E723D993C9CBEFE3D8FBE723F113F413E763D9B3C77BFDA3E3BBE703F133F443E7A3D9E3C3DBF303FAABD6D3F153F473E7E3DA13C2B3E613F8A3C6A3F173F4B3E813DA33C6C3F7C3FEF3D663F193F4E3E833DA63C543F7E3F5C3E633F1B3F513E853DA83CDABC663FA03E5F3F1D3F543E873DAB3C5BBF383FCF3E5B3F1F3F573E893DAE3C66BFF03EFD3E563F213F5B3E8B3DB03CEBBD2F3E143F523F233F5E3E8D3DB33C463F14BE283F4D3F253F613E8F3DB53C733FE4BE3B3F483F273F643E913DB83C823E34BF4B3F433F293F673E933DBB3C2DBF63BF5A3F3D3F2B3F6A3E953DBD3C7CBF7DBF663F383F2D3F6D3E973DC03CC7BE7DBF703F323F2E3F713E993DC23C113F64BF783F2C3F303F743E9B3DC53C803F35BF7D3F263F323F773E9E3DC73C043FE6BE803F203F343F7A3EA03DCA3CE3BE1ABE803F1A3F363F7D3EA23DCD3C7EBF293E7D3F133F383F803EA43DCF3C21BFED3E783F0C3F393F823EA63DD23CA03E373F713F053F3B3F833EA83DD43C783F663F673FFD3E3D3F853EAA3DD73C3C3F7D3F5B3FEF3E3F3F863EAC3DDA3C34BE7C3F4C3FE03E403F883EAE3DDC3C6CBF623F3C3FD23E423F8A3EB03DDF3C52BF313F2A3FC33E443F8B3EB23DE13C113DDD3E163FB43E453F8D3EB43DE43C5C3F053E003FA53E473F8E3EB63DE73C653F3EBED33E953E493F903EB83DE93CD93DF7BEA33E863E4A3F913EBA3DEC3C48BF3BBF643E6C3E4C3F933EBC3DEE3C73BF68BFFF3D4C3E4D3F943EBE3DF13C7BBE7EBFCB3C2C3E4F3F963EC03DF33C2F3F7BBF9ABD0C3E503F983EC23DF63C7C3F5FBF33BED83D523F993EC43DF93CC23E2DBF8BBE983D533F9B3EC63DFB3C13BFD3BEBCBE2E3D553F9C3EC83DFE3C80BFE0BDEABE333C563F9E3ECA3D003D02BF523E0BBFA9BC573F9F3ECC3D023DE73E003F20BF56BD593FA13ECE3D033D7F3F3F3F33BFACBD5A3FA23ED13D043D1F3F6A3F45BFECBD5B3FA43ED33D053DA5BE7F3F54BF16BE5D3FA53ED53D073D78BF7A3F61BF36BE5E3FA73ED73D083D3ABF5C3F6CBF56BE5F3FA83ED93D093D3D3E293F75BF75BE613FAA3EDB3D0B3D6D3FCA3E7BBF8ABE623FAB3EDD3D0C3D513FB53D7FBF9ABE633FAD3EDF3D0D3D35BD67BE80BFA9BE643FAF3EE13D0E3D5DBF04BF7FBFB8BE653FB03EE33D103D64BF42BF7BBFC7BE663FB23EE53D113DC7BD6CBF74BFD6BE683FB33EE73D123D493F7FBF6BBFE5BE693FB53EE93D143D723F79BF60BFF3BE6A3FB63EEB3D153D723E5ABF53BF01BF6B3FB83EED3D163D31BF25BF43BF08BF6C3FB93EEF3D183D7BBFC0BE32BF0EBF6D3FBB3EF13D193DBEBE8BBD1EBF15BF6E3FBC3EF33D1A3D153F7C3E09BF1CBF6F3FBE3EF53D1B3D803F093FE6BE22BF703FBF3EF73D1D3DFF3E453FB7BE28BF703FC13EF93D1E3DEBBE6E3F87BE2EBF713FC23EFB3D1F3D7FBF803F2ABE34BF723FC43EFD3D213D1EBF773F88BD3ABF733FC53EFF3D223DA93E573F0A3D3FBF743FC73E013E233D793F213F083E44BF753FC83E023E243D",
      torch_tensor_1_128_1_8_torch.bfloat16: "0x02000000803F803F803F803F803F803F803F803F0A3F733F7F3F803F803F803F803F803FD5BE4E3F7B3F7F3F803F803F803F803F7DBF153F753F7F3F803F803F803F803F27BF9A3E6C3F7E3F803F803F803F803F913E29BC613F7D3F803F803F803F803F763FA4BE533F7B3F803F803F803F803F413F19BF443F7A3F7F3F803F803F803F15BE52BF323F783F7F3F803F803F803F69BF75BF1F3F763F7F3F803F803F803F57BF80BF0A3F733F7F3F803F803F803F913B72BFE83E713F7E3F803F803F803F583F4BBFBA3E6E3F7E3F803F803F803F683F11BF893E6B3F7E3F803F803F803F0C3E90BE2E3E673F7D3F803F803F803F42BFFE3C913D643F7D3F803F803F803F75BFAE3EEFBC603F7D3F803F803F803F8DBE1E3F04BE5C3F7C3F803F803F803F293F553F69BE583F7C3F803F803F803F7D3F763FA6BE533F7B3F803F803F803FD13E803FD5BE4E3F7B3F7F3F803F803F0CBF703F01BF4A3F7A3F7F3F803F803F80BF483F17BF453F7A3F7F3F803F803F08BF0C3F2BBF3F3F793F7F3F803F803FD93E863E3DBF3A3F793F7F3F803F803F7E3F54BD4DBF343F783F7F3F803F803F263FB8BE5BBF2E3F773F7F3F803F803F96BE22BF67BF283F773F7F3F803F803F76BF57BF71BF223F763F7F3F803F803F40BF78BF79BF1C3F753F7F3F803F803F1E3E80BF7DBF153F753F7F3F803F803F6A3F6EBF80BF0F3F743F7F3F803F803F563F45BF80BF083F733F7F3F803F803F5ABC08BF7DBF013F723F7F3F803F803F59BF77BE78BFF33E713F7F3F803F803F67BF943D70BFE53E703F7E3F803F803F03BEC23E66BFD73E703F7E3F803F803F443F263F59BFC83E6F3F7E3F803F803F743F5A3F4ABFB93E6E3F7E3F803F803F893E793F3ABFAA3E6D3F7E3F803F803F2BBF7F3F27BF9A3E6C3F7E3F803F803F7DBF6C3F13BF8B3E6B3F7E3F803F803FCDBE413FFBBE763E6A3F7E3F803F803F0E3F033FCDBE563E693F7E3F803F803F803F633E9DBE373E683F7E3F803F803F063FBEBD58BE173E673F7D3F803F803FDDBECCBEE6BDED3D653F7D3F803F803F7EBF2ABF4BBCAD3D643F7D3F803F803F24BF5DBFB33D593D633F7D3F803F803F9A3E7ABF3F3EAE3C623F7D3F803F803F773F7FBF913E29BC613F7D3F803F803F3E3F6ABFC23E2CBD5F3F7D3F803F803F27BE3EBFF03E97BD5E3F7D3F803F803F6BBFFEBE0E3FD7BD5D3F7C3F803F803F54BF4EBE223F0CBE5C3F7C3F803F803FB53CE83D353F2CBE5A3F7C3F803F803F5A3FD53E473F4CBE593F7C3F803F803F663F2E3F563F6BBE583F7C3F803F803FF43D603F633F85BE563F7C3F803F803F45BF7B3F6D3F95BE553F7C3F803F803F74BF7E3F763FA4BE533F7B3F803F803F84BE683F7C3FB3BE523F7B3F803F803F2C3F3A3F7F3FC3BE503F7B3F803F803F7C3FF53E803FD1BE4F3F7B3F7F3F803FC93E393E7E3FE0BE4D3F7B3F7F3F803F10BF09BE7A3FEFBE4C3F7B3F7F3F803F80BFDFBE733FFDBE4A3F7A3F7F3F803F05BF32BF6A3F05BF493F7A3F7F3F803FE13E62BF5F3F0CBF473F7A3F7F3F803F7E3F7CBF513F13BF453F7A3F7F3F803F223F7DBF413F19BF443F7A3F7F3F803F9EBE65BF2F3F20BF423F7A3F7F3F803F78BF36BF1C3F26BF403F793F7F3F803F3CBFEBBE073F2CBF3F3F793F7F3F803F303E24BEE13E32BF3D3F793F7F3F803F6C3F1E3EB13E38BF3B3F793F7F3F803F533FE93E813E3DBF3A3F793F7F3F803FFEBC353F1D3E43BF383F783F7F3F803F5CBF653F5D3D48BF363F783F7F3F803F65BF7D3F3CBD4DBF343F783F7F3F803FE2BD7D3F15BE52BF323F783F7F3F803F473F633F79BE56BF313F783F7F3F803F733F333FAEBE5ABF2F3F773F7F3F803F803EE23EDDBE5FBF2D3F773F7F3F803F2EBF0F3E05BF62BF2B3F773F7F3F803F7CBF33BE1ABF66BF293F773F7F3F803FC4BEF2BE2EBF6ABF273F773F7F3F803F123F39BF40BF6DBF253F763F7F3F803F803F67BF50BF70BF233F763F7F3F803F033F7EBF5EBF72BF213F763F7F3F803FE5BE7CBF69BF75BF1F3F763F7F3F803F7FBF60BF73BF77BF1D3F753F7F3F803F20BF2FBF7ABF79BF1B3F753F7F3F803FA33ED8BE7EBF7BBF193F753F7F3F803F783FF5BD80BF7CBF173F753F7F3F803F3B3F483E7FBF7EBF153F753F7F3F803F39BEFB3E7CBF7FBF133F743F7F3F803F6DBF3D3F76BF7FBF113F743F7F3F803F52BF693F6EBF80BF0F3F743F7F3F803F233D7E3F64BF80BF0C3F743F7F3F803F5D3F7B3F57BF80BF0A3F733F7F3F803F643F5E3F48BF80BF083F733F7F3F803FD03D2B3F37BF7FBF063F733F7F3F803F48BFCF3E24BF7EBF043F733F7F3F803F72BFCA3D10BF7DBF023F723F7F3F803F77BE5DBEF3BE7CBFFF3E723F7F3F803F303F02BFC5BE7ABFFA3E723F7F3F803F7C3F40BF95BE79BFF63E713F7F3F803FC03E6BBF47BE76BFF13E713F7F3F803F14BF7FBFC3BD74BFED3E713F7E3F803F80BF79BF913B72BFE83E713F7E3F803F01BF5BBFD53D6FBFE43E703F7E3F803FE93E27BF503E6CBFDF3E703F7E3F803F7F3FC5BE993E69BFDA3E703F7E3F803F1F3FA0BDC93E65BFD63E703F7E3F803FA7BE713EF73E61BFD13E6F3F7E3F803F79BF073F113F5DBFCC3E6F3F7E3F803F39BF443F263F59BFC83E6F3F7E3F803F423E6D3F383F55BFC33E6E3F7E3F803F6E3F7F3F493F50BFBE3E6E3F7E3F803F503F783F583F4BBFBA3E6E3F7E3F803F47BD583F653F46BFB53E6D3F7E3F803F5EBF233F6F3F41BFB03E6D3F7E3F803F63BFBB3E773F3CBFAB3E6D3F7E3F803FBEBD6C3D7C3F36BFA63E6D3F7E3F803F4A3F83BE7F3F30BFA13E6C3F7E3F803F723F0BBF803F2ABF9D3E6C3F7E3F803F6E3E47BF7E3F24BF983E6C3F7E3F803F"
    }
  }
#-}
