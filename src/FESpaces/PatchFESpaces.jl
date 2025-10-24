
# This FESpace preserves the order of the patch dofs between the original space 
# and this one. 
function FESpaceWithoutBCs(space::SingleFieldFESpace)
  nfree = num_free_dofs(space)
  ndir = num_dirichlet_dofs(space)
  free_ids = Int32(ndir+1):Int32(nfree+ndir)
  dir_ids = Int32(ndir):Int32(-1):Int32(1) # Reverse order
  return reindex_free_and_dirichlet_dof_ids(space,free_ids,dir_ids)
end

FESpaceWithoutBCs(space::ConstantFESpace) = space
FESpaceWithoutBCs(space::TrialFESpace) = FESpaceWithoutBCs(space.space)

# Index mapping from global to local (consecutive) indices
struct Global2Local
  g2l::Dict{Int32,Int32}
end

function Global2Local(globalIds::Vector{Int32})
  Global2Local(Dict(g => l for (l,g) in enumerate(globalIds)))
end

function (g2l::Global2Local)(gidx)
  get(g2l.g2l,gidx,missing)
end

function PatchFESpaces(space::SingleFieldFESpace,ptopo::PatchTopology)
  vector_type = get_vector_type(space)

  nfree = num_free_dofs(space)
  patch_cell_ids = Geometry.get_patch_cells(ptopo)
  patch_edge_ids = Geometry.get_patch_facets(ptopo)
  patch_vertex_ids = Geometry.get_patch_faces(ptopo,0)

  #=
  function get_offsets(ptopo::PatchTopology{Dc}) where Dc
    offsets = zeros(Int, Dc)
    temp = 0
    for d in 1:Dc
      offsets[d] = temp
      temp += num_faces(ptopo,d)
    end
    return offsets
  end
  =#

  global_cell_dof_ids = get_cell_dof_ids(space)

  model = get_background_model(get_triangulation(space))

  global_fe_basis = get_fe_basis(space)
  global_cell_basis = get_data(global_fe_basis)

  global_fe_dof_basis = get_fe_dof_basis(space)
  global_cell_dofs = get_data(global_fe_dof_basis)

  basis_style = BasisStyle(global_fe_basis)

  # global constraints
  ndirichlet = num_dirichlet_dofs(space)
  dirichlet_dof_tag = get_dirichlet_dof_tag(space)
  ntags = num_dirichlet_tags(space)

  # information for local geometry
  topo = get_grid_topology(model)
  labels = get_face_labeling(model)
  is_global_boundary = get_face_mask(labels, "boundary", 1)
  edge_to_vertex = Geometry.get_faces(topo, 1, 0)
  cell_to_edge = Geometry.get_faces(topo, 2, 1)
  vertex_to_edge = Geometry.get_faces(topo, 0, 1)

  D = Geometry.num_cell_dims(ptopo)
   # TODO this is kinda hacky, add at least a check here
  d_to_ctype_to_ldface_to_own_ldofs = space.metadata.d_ctype_ldface_own_ldofs

  npatches = length(patch_cell_ids)
  spaces = Vector{UnconstrainedFESpace}(undef,npatches)
  for patch in 1:npatches
    local_cell_ids = patch_cell_ids[patch]

    trian = Triangulation(model, collect(local_cell_ids))

    local_fe_basis = SingleFieldFEBasis(
      getindex(global_cell_basis, local_cell_ids),
      trian,
      basis_style,
      ReferenceDomain()
    )

    local_fe_dof_basis = CellDof(
      getindex(global_cell_dofs, local_cell_ids),
      trian,
      ReferenceDomain()
      )

    # identify interior boundary edges
    # TODO assume no refinement, need to go up hierarchy eventually
    boundary_edges = Vector{Int32}()
    for edge in patch_edge_ids[patch]
      adjacent_vertices = edge_to_vertex[edge]
      if patch ∉ adjacent_vertices && ~is_global_boundary[edge]
          push!(boundary_edges, edge)
      end
    end

    boundary_vertices = Vector{Int32}()
    for vtx in patch_vertex_ids[patch]
      if all([e in boundary_edges for e in vertex_to_edge[vtx]])
        push!(boundary_vertices, vtx)
      end
    end

    # Global/Local face identification
    d_to_dface_ids = [Geometry.get_patch_faces(ptopo,d)[patch] for d in 0:D]
    
    d_to_g2l = [Global2Local(ids) for ids in d_to_dface_ids]
    d_to_l2g = [Reindex(ids) for ids in d_to_dface_ids]

    d_to_num_dfaces = length.(d_to_dface_ids)
    n_faces = sum(d_to_num_dfaces)
    cell_to_ctype = getindex(get_cell_type(topo), local_cell_ids)

    # mappings in local enumeration
    d_to_cell_to_dfaces = [Table([map(d_to_g2l[d+1], dfaces)
      for dfaces in getindex(get_faces(topo,D,d),local_cell_ids)])
      for d in 0:D]
    
    d_to_dface_to_cells = [ getindex(get_faces(topo,d,D),d_to_dface_ids[d+1]) for d in 0:D]
    d_to_dface_to_cells = [ Table([map(d_to_g2l[D+1], filter(i -> i in local_cell_ids, cells))
     for cells in d_to_dface_to_cells[d+1]])
     for d in 0:D ]
    
    # TODO pull out into function
    d_to_offset = cumsum(d_to_num_dfaces) .- d_to_num_dfaces

    # generate patch-local dof mappings
    face_to_own_dofs, ntotal, d_to_dface_to_cell, d_to_dface_to_ldface =  FESpaces._generate_face_to_own_dofs(
      n_faces,
      cell_to_ctype,
      d_to_cell_to_dfaces,
      d_to_dface_to_cells,
      d_to_offset,
      d_to_ctype_to_ldface_to_own_ldofs)

    # extract patch-local dofs (global numbering)
    local_cell_dof_ids = getindex(global_cell_dof_ids, local_cell_ids)
    patch_dof_ids = sort(unique(vcat(local_cell_dof_ids...)))

    # create tags
    # TODO slightly hacky and inefficient
    # TODO missing constrained vertices
    d_to_dface_to_tag = Array{Vector{Int}}(undef,D+1)
    for d in 0:D
      if d != D
        # set global boundary tags
        d_to_dface_to_tag[d+1] = getindex(get_isboundary_face(topo,d), d_to_dface_ids[d+1])
      else
        d_to_dface_to_tag[d+1] = fill(Int(UNSET), d_to_num_dfaces[d+1])
      end
      if d == 0
        # internal vertices
        # TODO vectorize
        for vi in map(d_to_g2l[d+1],boundary_vertices)
          d_to_dface_to_tag[d+1][vi] = 2
        end
      end
      if d == D-1
        # internal edges
        # TODO vectorize
        for li in map(d_to_g2l[d+1],boundary_edges)
          d_to_dface_to_tag[d+1][li] = 2
        end
      end
    end

    # create local dof numbering
    nfree, ndirichlet, dirichlet_dof_tag = FESpaces._split_face_own_dofs_into_free_and_dirichlet_generic!(
      face_to_own_dofs,
      d_to_offset,
      d_to_dface_to_tag)

    # renumbering into patch-local indices
    g2l_dof_numbering = Dict{Int32,Int32}()
    l2g_dof_numbering = Dict{Int32,Int32}()
    for d in 0:D
      offset = d_to_offset[d+1]
      for face in 1:d_to_num_dfaces[d+1]
        dface = offset+face
        pdofs = face_to_own_dofs[dface]
        cell = d_to_dface_to_cell[d+1][face]
        ldface = d_to_dface_to_ldface[d+1][face]
        ctype = cell_to_ctype[cell]
        own_ldofs = d_to_ctype_to_ldface_to_own_ldofs[d+1][ctype][ldface]
        own_gdofs = getindex(local_cell_dof_ids[cell],own_ldofs)
        merge!(l2g_dof_numbering, Dict(zip(pdofs,own_gdofs)))
        merge!(g2l_dof_numbering, Dict(zip(own_gdofs,pdofs)))
      end
    end

    # get additional info for local space construction
    patch_cell_dof_ids = lazy_map(I -> map(i->get(g2l_dof_numbering,i,missing),I),local_cell_dof_ids)
    cell_has_dirichlet_dof = collect(Bool,lazy_map(I -> any(i -> i < 0, I), patch_cell_dof_ids))
    dirichlet_cell_ids = collect(Int32,findall(cell_has_dirichlet_dof))
    ntags = length(dirichlet_dof_tag)

    metadata = (ptopo, patch, l2g_dof_numbering, g2l_dof_numbering)

    localSpace =
    UnconstrainedFESpace(
      vector_type,
      nfree,
      ndirichlet,
      patch_cell_dof_ids,
      local_fe_basis,
      local_fe_dof_basis,
      cell_has_dirichlet_dof,
      dirichlet_dof_tag,
      dirichlet_cell_ids,
      ntags,
      metadata
    )

    spaces[patch] = localSpace
  end
  return spaces
end

function PatchFESpace(space::SingleFieldFESpace, ptopo::PatchTopology)
  vector_type = get_vector_type(space)

  nfree = num_free_dofs(space)
  patch_dof_ids = generate_patch_dof_ids(space,ptopo)

  model = get_background_model(get_triangulation(space))
  ptrian = PatchTriangulation(model,ptopo)
  cell_dof_ids = Table(lazy_map(Broadcasting(Reindex(patch_dof_ids)), ptrian.glue.tface_to_patch))

  Dc = num_cell_dims(model)
  cell_to_ndofs = collect(lazy_map(length,cell_dof_ids))
  type_to_ndofs = unique(cell_to_ndofs)
  cell_to_type = collect(Int8,indexin(cell_to_ndofs,type_to_ndofs))
  type_to_basis = [MockFieldArray(zeros(Float64,ndofs)) for ndofs in type_to_ndofs]
  type_to_dofs = [ReferenceFEs.MockDofBasis(zeros(VectorValue{Dc,Float64},ndofs)) for ndofs in type_to_ndofs] 
  cell_shapefuns = expand_cell_data(type_to_basis,cell_to_type)
  cell_dof_basis = expand_cell_data(type_to_dofs,cell_to_type)
  fe_basis = GenericCellField(cell_shapefuns,ptrian,ReferenceDomain())
  fe_dof_basis = CellDof(cell_dof_basis,ptrian,ReferenceDomain())

  ndirichlet = num_dirichlet_dofs(space)
  dirichlet_dof_tag = get_dirichlet_dof_tag(space)
  ntags = num_dirichlet_tags(space)

  cell_is_dirichlet = collect(Bool,lazy_map(I -> any(i -> i < 0, I), cell_dof_ids))
  dirichlet_cells = collect(Int32,findall(cell_is_dirichlet))

  metadata = ptopo
  UnconstrainedFESpace(
    vector_type,
    nfree,
    ndirichlet,
    cell_dof_ids,
    fe_basis,
    fe_dof_basis,
    cell_is_dirichlet,
    dirichlet_dof_tag,
    dirichlet_cells,
    ntags,
    metadata
  )
end

function PatchFESpace(space::TrialFESpace,ptopo::PatchTopology)
  TrialFESpace(space.dirichlet_values,PatchFESpace(space.space,ptopo))
end

function generate_patch_dof_ids(
  space::SingleFieldFESpace, ptopo::PatchTopology
)
  trian = get_triangulation(space)
  Df = num_cell_dims(trian)
  glue = get_glue(trian,Val(Df))
  patch_to_lpface_to_mface = get_patch_faces(ptopo,Df)

  tface_to_dofs = get_cell_dof_ids(space)
  mface_to_dofs = extend(tface_to_dofs,glue.mface_to_tface)
  patch_to_dofs = Arrays.merge_entries(
    mface_to_dofs,patch_to_lpface_to_mface;
    acc = SortedSet{Int32}()
  )
  return patch_to_dofs
end

function generate_pface_to_pdofs(
  space::SingleFieldFESpace, ptopo::PatchTopology
)
  Df = num_cell_dims(get_triangulation(space))
  pface_to_dofs = get_cell_dof_ids(space)
  patch_to_dofs = generate_patch_dof_ids(space,ptopo)
  pface_to_patch = Geometry.get_pface_to_patch(ptopo,Df)
  pface_to_pdofs = find_local_index(pface_to_dofs,pface_to_patch,patch_to_dofs)
  return pface_to_pdofs
end

function generate_pface_to_pdofs(
  space::SingleFieldFESpace, ptrian::PatchTriangulation,
)
  pface_to_dofs = get_cell_dof_ids(space,ptrian)
  patch_to_dofs = generate_patch_dof_ids(space,ptrian.ptopo)
  pface_to_patch = ptrian.glue.tface_to_patch
  pface_to_pdofs = find_local_index(pface_to_dofs,pface_to_patch,patch_to_dofs)
  return pface_to_pdofs
end

# Changes of domain

function CellData.change_domain(
  a::CellField,strian::PatchTriangulation,::ReferenceDomain,ttrian::PatchTriangulation,::ReferenceDomain
)
  if strian === ttrian
    return a
  end
  if is_change_possible(strian.trian,ttrian.trian)
    b = change_domain(a,strian.trian,ReferenceDomain(),ttrian.trian,ReferenceDomain())
    return CellData.similar_cell_field(b,CellData.get_data(b),ttrian,ReferenceDomain())
  end
  @assert num_cell_dims(strian) == num_cell_dims(ttrian)
  sglue = Geometry.get_patch_glue(strian)
  tglue = Geometry.get_patch_glue(ttrian)
  @check is_change_possible(sglue,tglue)
  return CellData.change_domain_ref_ref(a,ttrian,sglue,tglue)
end

function CellData.change_domain(
  a::CellField,strian::PatchTriangulation,::PhysicalDomain,ttrian::PatchTriangulation,::PhysicalDomain
)
  if strian === ttrian
    return a
  end
  if is_change_possible(strian.trian,ttrian.trian)
    b = change_domain(a,strian.trian,PhysicalDomain(),ttrian.trian,PhysicalDomain())
    return CellData.similar_cell_field(b,CellData.get_data(b),ttrian,PhysicalDomain())
  end
  @assert num_cell_dims(strian) == num_cell_dims(ttrian)
  sglue = Geometry.get_patch_glue(strian)
  tglue = Geometry.get_patch_glue(ttrian)
  @check is_change_possible(sglue,tglue)
  return CellData.change_domain_phys_phys(a,ttrian,sglue,tglue)
end

function get_cell_fe_data(fun,f::SingleFieldFESpace,ttrian::PatchTriangulation)
  strian = get_triangulation(f)
  get_cell_fe_data(fun,f,strian,ttrian)
end

function get_cell_fe_data(fun,f,strian::Triangulation,ttrian::PatchTriangulation)
  sface_to_data = fun(f)
  if strian === ttrian
    return sface_to_data
  end
  @check is_change_possible(strian,ttrian)
  D = num_cell_dims(strian)
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))
  get_cell_fe_data(fun,sface_to_data,sglue,tglue)
end

function get_cell_fe_data(fun,f,strian::PatchTriangulation,ttrian::PatchTriangulation)
  sface_to_data = fun(f)
  if strian === ttrian
    return sface_to_data
  end
  if is_change_possible(strian.trian,ttrian.trian)
    D = num_cell_dims(strian)
    sglue = get_glue(strian,Val(D))
    tglue = get_glue(ttrian,Val(D))
    return get_cell_fe_data(fun,sface_to_data,sglue,tglue)
  end
  @assert num_cell_dims(strian) == num_cell_dims(ttrian)
  sglue = Geometry.get_patch_glue(strian)
  tglue = Geometry.get_patch_glue(ttrian)
  @check is_change_possible(sglue,tglue)
  return get_cell_fe_data(fun,sface_to_data,sglue,tglue)
end

############################################################################################

function PatchFESpace(model::DiscreteModel, ptopo::PatchTopology, args...; kwargs...)
  PatchFESpace(
    model, PatchTriangulation(model,ptopo), args...; kwargs...
  )
end

function PatchFESpace(trian::PatchTriangulation, args...; kwargs...)
  PatchFESpace(
    get_background_model(trian), trian, args...; kwargs...
  )
end

function PatchFESpace(
  model::DiscreteModel,trian::PatchTriangulation,::Type{T},order::Integer;
  space = :P, 
  vector_type = nothing,
  hierarchical = false, 
  orthonormal = false,
  local_kernel = nothing,
  kwargs...
) where T
  patch_grid = Geometry.bounding_box_grid(model, trian; δmin=0.1)
  patch_shapefuns, domain_style = get_polytopal_cell_shapefuns(
    patch_grid, T, order; space, hierarchical, orthonormal, local_kernel, domain_style = PhysicalDomain()
  )
  @check isa(domain_style, PhysicalDomain)
  vtype = ifelse(!isnothing(vector_type),vector_type,Vector{_dof_type(T)})
  return PatchFESpace(
    vtype, model, trian, patch_grid, patch_shapefuns, order, domain_style; kwargs...
  )
end

function PatchFESpace(
  vector_type::Type,
  model::DiscreteModel,
  trian::PatchTriangulation,
  patch_grid::Grid,
  patch_shapefuns::AbstractArray,
  order::Integer,
  domain_style::DomainStyle;
  labels = get_face_labeling(model),
  dirichlet_tags = Int[],
  dirichlet_masks = nothing,
)
  patch_to_faces = trian.glue.patch_to_tfaces
  face_to_patch = trian.glue.tface_to_patch

  npatches = num_cells(patch_grid)
  patch_to_ctype = Base.OneTo(npatches)
  ctype_to_ndofs = map(length, patch_shapefuns)

  ntags = length(dirichlet_tags)
  if ntags != 0
    face_to_tag = get_face_tag_index(labels,dirichlet_tags,Df)
    patch_to_tag = zeros(Int32, npatches)
    patch_is_dirichlet = zeros(Bool, npatches)
    for (patch,faces) in enumerate(patch_to_faces)
      tag = only(unique(face_to_tag[faces]))
      patch_to_tag[patch] = tag
      patch_is_dirichlet[patch] = !iszero(tag)
    end
    
    ctype_to_ldof_to_comp = lazy_map(n -> ones(Int16,n), ctype_to_ndofs)
    patch_dof_ids, nfree, ndir, dirichlet_dof_tag, dirichlet_patches = FESpaces.compute_discontinuous_cell_dofs(
      patch_to_ctype, ctype_to_ndofs, ctype_to_ldof_to_comp, patch_to_tag, dirichlet_masks
    )
  else
    ndir = 0
    dirichlet_dof_tag = Int8[]
    dirichlet_patches = Int32[]
    patch_is_dirichlet = fill(false,npatches)
    patch_dof_ids, nfree = FESpaces.compute_discontinuous_cell_dofs(patch_to_ctype,ctype_to_ndofs)
  end

  cell_shapefuns = lazy_map(Reindex(patch_shapefuns), face_to_patch)
  fe_basis = SingleFieldFEBasis(cell_shapefuns, trian, TestBasis(), domain_style)

  cell_dof_ids = Table(lazy_map(Broadcasting(Reindex(patch_dof_ids)), face_to_patch))
  cell_is_dirichlet = patch_is_dirichlet[face_to_patch]
  dirichlet_cells = reduce(vcat, (patch_to_faces[patch] for patch in dirichlet_patches); init = Int32[])

  metadata = nothing
  return PolytopalFESpace(
    vector_type,nfree,ndir,cell_dof_ids,fe_basis,
    cell_is_dirichlet,dirichlet_dof_tag,dirichlet_cells,ntags,
    order,metadata
  )
end
