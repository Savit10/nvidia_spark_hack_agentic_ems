import cudf, cuml, cugraph, cupy

print("cudf   :", cudf.__version__)
print("cuml   :", cuml.__version__)
print("cugraph:", cugraph.__version__)
print("cupy   :", cupy.__version__)

print("GPU     :", cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())

df = cudf.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
df["c"] = df["a"] * df["b"]
print("cuDF sum of c =", int(df["c"].sum()), "(expected 550)")
print("OK - RAPIDS is running on the GPU")
