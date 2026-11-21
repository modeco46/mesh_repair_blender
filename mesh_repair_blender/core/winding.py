"""Evaluate bounded CUDA batches and assemble one continuous isosurface."""
import time


def winding_slabs(mesh, mm, winding, params, completed, *, sharpen=False, budget=2_000_000, deadline=None):
    step = params.voxelSize
    grid = mm.calcOriginAndDimensions(mesh.computeBoundingBox(), step)
    dims = grid.dimensions
    spacing = mm.Vector3f.diagonal(step)
    basis = mm.Matrix3f.scale(spacing)
    first_center = grid.origin + spacing*.5
    options = mm.DistanceToMeshOptions()
    options.maxDistSq = (1.001*step)**2
    options.nullOutsideMinMax = not params.closeHolesInHoleWindingNumber
    options.windingNumberThreshold = params.windingNumberThreshold
    options.windingNumberBeta = params.windingNumberBeta
    marching = mm.MarchingCubesParams()
    marching.origin, marching.iso, marching.lessInside = grid.origin, 0, True
    if sharpen:
        voxel_per_face = mm.Vector_VoxelId_FaceId()
        grid_to_mesh = mm.AffineXf3f()
        marching.outVoxelPerFaceMap, marching.outGridToMeshXf = voxel_per_face, grid_to_mesh
    mesher = mm.MarchingCubesByParts(dims, marching)
    plane = dims.x*dims.y
    depth = max(2, budget//plane)
    overlap = None
    blocks = 0
    progress = {'grid': [dims.x, dims.y, dims.z], 'total_layers': dims.z, 'block_layers': min(depth, dims.z),
                'completed_layers': 0, 'completed_blocks': 0}
    completed(progress.copy())
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Истёк лимит времени уточнения winding.')
        z = mesher.nextZ()
        size = mm.Vector3i(dims.x, dims.y, min(depth, dims.z-z))
        volume = mm.SimpleVolume()
        volume.dims, volume.voxelSize = size, spacing
        data = volume.data.vec_
        # The CUDA binding writes to res.data(); it does not resize the vector.
        data.resize(plane*size.z)
        origin = first_center + mm.Vector3f(0, 0, z*step)
        # Both native calls complete on this worker before notifying Python.
        # Never call Python from MeshLib's asynchronous GPU producer thread.
        winding.calcFromGridWithDistances(data, size, mm.AffineXf3f(basis, origin), options,
                                          mm.func_bool_from_float())
        if overlap is not None:
            # Independently shifted float grid origins can differ by one ULP.
            # Marching cubes must see identical values on both sides of a seam.
            data[:plane] = overlap
        end = z+size.z
        overlap = data[-plane:] if end < dims.z else None
        mesher.addPart(volume)
        blocks += 1
        progress.update(completed_layers=end, completed_blocks=blocks)
        completed(progress.copy())
        if end == dims.z:
            break
    result = mm.Mesh.fromTriMesh(mesher.finalize())
    if sharpen:
        detail = mm.SharpenMarchingCubesMeshSettings()
        detail.dims, detail.gridToMeshXf = dims, grid_to_mesh
        detail.voxelClamp = params.voxelClamp
        detail.minNewVertDev = step*params.minNewVertDev
        detail.maxNewRank2VertDev = step*params.maxNewRank2VertDev
        detail.maxNewRank3VertDev = step*params.maxNewRank3VertDev
        detail.maxOldVertPosCorrection = step*params.maxOldVertPosCorrection
        mm.sharpenMarchingCubesMesh(mm.MeshPart(mesh), result, voxel_per_face, detail)
    return result
