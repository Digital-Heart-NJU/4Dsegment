// Exact intersection geometry for Step 08. Tetrahedral tags stay in Python.
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/intersections.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/boost/graph/helpers.h>
#include <boost/optional.hpp>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;
using Face = Mesh::Face_index;
using Vertex = Mesh::Vertex_index;
using Tree = CGAL::AABB_tree<CGAL::AABB_traits<Kernel,CGAL::AABB_face_graph_triangle_primitive<Mesh>>>;
namespace PMP = CGAL::Polygon_mesh_processing;

struct RegionVisitor : PMP::Corefinement::Default_visitor<Mesh> {
  std::map<const Mesh*,int> current;
  static Mesh::Property_map<Face,int> map(Mesh& mesh) {
    return mesh.add_property_map<Face,int>("f:region",0).first;
  }
  void before_subface_creations(Face face,Mesh& mesh) { current[&mesh]=map(mesh)[face]; }
  void after_subface_created(Face face,Mesh& mesh) { map(mesh)[face]=current[&mesh]; }
  void after_face_copy(Face from,const Mesh& source,Face to,Mesh& target) {
    auto property=source.property_map<Face,int>("f:region");
    map(target)[to]=property.second?property.first[from]:0;
  }
};

static const int tetra_faces[4][3] = {{0,2,1},{0,1,3},{0,3,2},{1,2,3}};

template<class T> T read_scalar(std::istream& stream) {
  T value;
  stream.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!stream) throw std::runtime_error("Truncated exact-clip input.");
  return value;
}

template<class T> void write_scalar(std::ostream& stream, const T& value) {
  stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

Point read_point(std::istream& stream) {
  double x = read_scalar<double>(stream);
  double y = read_scalar<double>(stream);
  double z = read_scalar<double>(stream);
  return Point(x, y, z);
}

Mesh read_mesh(std::istream& stream) {
  const auto nv = read_scalar<std::int64_t>(stream);
  const auto nf = read_scalar<std::int64_t>(stream);
  Mesh mesh;
  std::vector<Vertex> vertices;
  for (std::int64_t i = 0; i < nv; ++i) vertices.push_back(mesh.add_vertex(read_point(stream)));
  for (std::int64_t i = 0; i < nf; ++i) {
    const auto a = read_scalar<std::int64_t>(stream);
    const auto b = read_scalar<std::int64_t>(stream);
    const auto c = read_scalar<std::int64_t>(stream);
    if (a < 0 || b < 0 || c < 0 || a >= nv || b >= nv || c >= nv ||
        mesh.add_face(vertices[a], vertices[b], vertices[c]) == Mesh::null_face())
      throw std::runtime_error("Invalid or inconsistently oriented cutter triangle.");
  }
  if (!CGAL::is_closed(mesh)) throw std::runtime_error("The cutter is not closed.");
  return mesh;
}

Mesh tetra_mesh(const std::array<Point,4>& points) {
  Mesh mesh;
  std::array<Vertex,4> vertices;
  for (int i=0; i<4; ++i) vertices[i]=mesh.add_vertex(points[i]);
  for (const auto& face : tetra_faces)
    mesh.add_face(vertices[face[0]],vertices[face[1]],vertices[face[2]]);
  return mesh;
}

void label_mesh(Mesh& mesh,int label) {
  auto property=RegionVisitor::map(mesh);
  for(Face f:mesh.faces()) property[f]=label;
}

std::array<bool,4> booleans(Mesh& a,Mesh& b,Mesh* united,Mesh* inside,Mesh* outside) {
  RegionVisitor visitor;
  std::array<boost::optional<Mesh*>,4> results;
  if(united) results[PMP::Corefinement::UNION]=united;
  if(inside) results[PMP::Corefinement::INTERSECTION]=inside;
  if(outside) results[PMP::Corefinement::TM1_MINUS_TM2]=outside;
  return PMP::corefine_and_compute_boolean_operations(a,b,results,
      CGAL::parameters::visitor(visitor).throw_on_self_intersection(true));
}

Mesh original_patch(const Mesh& source) {
  Mesh patch;
  std::map<Point,Vertex,Kernel::Less_xyz_3> ids;
  auto property=source.property_map<Face,int>("f:region").first;
  for(Face f:source.faces()) if(property[f]==1) {
    std::vector<Vertex> triangle;
    for(Vertex v:CGAL::vertices_around_face(source.halfedge(f),source)) {
      Point p=source.point(v);
      auto found=ids.find(p);
      if(found==ids.end()) found=ids.emplace(p,patch.add_vertex(p)).first;
      triangle.push_back(found->second);
    }
    if(patch.add_face(triangle)==Mesh::null_face())
      throw std::runtime_error("The exact basal patch is not an oriented surface.");
  }
  return patch;
}

Mesh extend_cap(const Mesh& protected_patch,const Mesh& basal,const Kernel::Vector_3& delta) {
  Mesh extended;
  std::map<Point,Vertex,Kernel::Less_xyz_3> ids;
  auto vertex=[&](const Point& p) {
    auto found=ids.find(p);
    if(found==ids.end()) found=ids.emplace(p,extended.add_vertex(p)).first;
    return found->second;
  };
  auto add=[&](const Point& a,const Point& b,const Point& c) {
    if(extended.add_face(vertex(a),vertex(b),vertex(c))==Mesh::null_face())
      throw std::runtime_error("Exact basal extension has inconsistent face connectivity.");
  };
  for(const Mesh* source:{&protected_patch,&basal}) {
    for(Face f:source->faces()) {
      std::vector<Point> p;
      for(Vertex v:CGAL::vertices_around_face(source->halfedge(f),*source))
        p.push_back(source->point(v)+(source==&basal?delta:CGAL::NULL_VECTOR));
      if(p.size()!=3) throw std::runtime_error("The basal patch is not triangulated.");
      add(p[0],p[1],p[2]);
    }
  }
  for(auto h:basal.halfedges()) if(!basal.is_border(h)&&basal.is_border(basal.opposite(h))) {
    Point a=basal.point(basal.source(h)),b=basal.point(basal.target(h));
    add(a,b,b+delta);
    add(a,b+delta,a+delta);
  }
  if(!CGAL::is_closed(extended)||PMP::does_self_intersect(extended))
    throw std::runtime_error("The exact locally extended cavity is open or self-intersecting.");
  return extended;
}

void write_mesh_binary(std::ostream& out,const Mesh& mesh) {
  write_scalar<std::int64_t>(out,num_vertices(mesh));
  write_scalar<std::int64_t>(out,num_faces(mesh));
  std::map<Vertex,std::int64_t> ids;
  for(Vertex v:mesh.vertices()) {
    ids[v]=ids.size();
    const Point& p=mesh.point(v);
    for(auto value:{p.x(),p.y(),p.z()}) write_scalar<double>(out,CGAL::to_double(value));
  }
  for(Face f:mesh.faces())
    for(Vertex v:CGAL::vertices_around_face(mesh.halfedge(f),mesh))
      write_scalar<std::int64_t>(out,ids.at(v));
}

void write_exact_mesh(const std::string& filename,const Mesh& mesh) {
  std::ofstream out(filename);
  out<<"EXACT_BASAL_CUTTER_1\n"<<num_vertices(mesh)<<" "<<num_faces(mesh)<<"\n";
  std::map<Vertex,std::int64_t> ids;
  for(Vertex v:mesh.vertices()) {
    ids[v]=ids.size();
    const auto& p=mesh.point(v);
    out<<CGAL::exact(p.x())<<" "<<CGAL::exact(p.y())<<" "<<CGAL::exact(p.z())<<"\n";
  }
  auto region=mesh.property_map<Face,int>("f:region").first;
  for(Face f:mesh.faces()) {
    for(Vertex v:CGAL::vertices_around_face(mesh.halfedge(f),mesh)) out<<ids.at(v)<<" ";
    out<<region[f]<<"\n";
  }
  if(!out) throw std::runtime_error("Could not write exact rational cutter.");
}

Mesh read_exact_mesh(const std::string& filename) {
  std::ifstream in(filename);
  std::string header; std::int64_t nv,nf;
  in>>header>>nv>>nf;
  if(header!="EXACT_BASAL_CUTTER_1") throw std::runtime_error("Invalid exact cutter format.");
  Mesh mesh;std::vector<Vertex> vertices;
  for(std::int64_t i=0;i<nv;++i) {
    Kernel::FT x,y,z;in>>x>>y>>z;
    vertices.push_back(mesh.add_vertex(Point(x,y,z)));
  }
  auto region=RegionVisitor::map(mesh);
  for(std::int64_t i=0;i<nf;++i) {
    std::int64_t a,b,c;int role;in>>a>>b>>c>>role;
    Face f=mesh.add_face(vertices.at(a),vertices.at(b),vertices.at(c));
    if(f==Mesh::null_face()) throw std::runtime_error("Invalid exact cutter connectivity.");
    region[f]=role;
  }
  if(!in) throw std::runtime_error("Truncated exact rational cutter.");
  return mesh;
}

void prepare(const char*input,const char*artifacts,const char*exact) {
  std::ifstream in(input,std::ios::binary);
  Mesh outer=read_mesh(in);
  std::array<Mesh,2> original={read_mesh(in),read_mesh(in)};
  std::array<Mesh,2> basal,cavities;
  for(int chamber=0;chamber<2;++chamber) {
    double dx=read_scalar<double>(in),dy=read_scalar<double>(in),dz=read_scalar<double>(in);
    Kernel::Vector_3 delta(dx,dy,dz);
    Mesh endo=original[chamber],back=outer;
    label_mesh(endo,1);label_mesh(back,0);
    for(Vertex v:back.vertices()) back.point(v)=back.point(v)-delta;
    Mesh inside,outside;
    auto success=booleans(endo,back,nullptr,&inside,&outside);
    if(!success[PMP::Corefinement::INTERSECTION]||!success[PMP::Corefinement::TM1_MINUS_TM2])
      throw std::runtime_error("Exact rim split is not manifold.");
    basal[chamber]=original_patch(outside);
    Mesh protected_patch=original_patch(inside);
    if(num_faces(basal[chamber])==0||num_faces(protected_patch)==0)
      throw std::runtime_error("The exact rim does not split the endocardium into two patches.");
    Mesh extended=extend_cap(protected_patch,basal[chamber],delta),fixed=original[chamber];
    success=booleans(extended,fixed,nullptr,nullptr,&cavities[chamber]);
    if(!success[PMP::Corefinement::TM1_MINUS_TM2])
      throw std::runtime_error("The exact added basal cavity is not manifold.");
    label_mesh(cavities[chamber],chamber+1);
    std::cerr<<"[CGAL prepare] chamber "<<chamber<<": "<<num_faces(basal[chamber])<<" basal triangles\n";
  }
  Mesh cutter;
  auto success=booleans(cavities[0],cavities[1],&cutter,nullptr,nullptr);
  if(!success[PMP::Corefinement::UNION])
    throw std::runtime_error("The exact LV/RV cavity union is not manifold.");
  write_exact_mesh(exact,cutter);
  Mesh restored=read_exact_mesh(exact);
  if(PMP::volume(restored)!=PMP::volume(cutter))
    throw std::runtime_error("Exact cutter serialization changed the enclosed volume.");
  std::ofstream out(artifacts,std::ios::binary);
  write_mesh_binary(out,cutter);
  auto regions=cutter.property_map<Face,int>("f:region").first;
  for(Face f:cutter.faces()) write_scalar<std::int64_t>(out,regions[f]);
  for(const Mesh& patch:basal) write_mesh_binary(out,patch);
}

Kernel::FT positive_volume(const Mesh& mesh) {
  if (num_faces(mesh)==0) return Kernel::FT(0);
  if (!CGAL::is_closed(mesh)) throw std::runtime_error("An exact Boolean boundary is not closed.");
  auto value=PMP::volume(mesh);
  if (value < 0) throw std::runtime_error("An exact Boolean has negative oriented volume.");
  return value;
}

struct Registry {
  std::vector<Point> points;
  std::map<Point,std::int64_t,Kernel::Less_xyz_3> ids;
  std::int64_t add(const Point& point) {
    const auto found=ids.find(point);
    if(found!=ids.end()) return found->second;
    auto index=static_cast<std::int64_t>(points.size());
    points.push_back(point);
    ids.emplace(point,index);
    return index;
  }
};

struct OutputPiece {
  std::int64_t parent;
  int state;
  std::array<std::vector<std::array<std::int64_t,4>>,2> faces;
};

bool triangle_crosses_interior(Face face,const Mesh& mesh,const std::array<Point,4>& tet) {
  std::vector<Point> polygon;
  for(Vertex v:CGAL::vertices_around_face(mesh.halfedge(face),mesh)) polygon.push_back(mesh.point(v));
  std::array<Kernel::Plane_3,4> planes;
  for(int i=0;i<4;++i) {
    planes[i]=Kernel::Plane_3(tet[tetra_faces[i][0]],tet[tetra_faces[i][1]],tet[tetra_faces[i][2]]);
    auto value=[&](const Point& p) {return planes[i].a()*p.x()+planes[i].b()*p.y()+planes[i].c()*p.z()+planes[i].d();};
    std::vector<Point> clipped;
    for(std::size_t j=0;j<polygon.size();++j) {
      const Point& a=polygon[j]; const Point& b=polygon[(j+1)%polygon.size()];
      auto va=value(a),vb=value(b);
      if(va<=0) clipped.push_back(a);
      if((va<0&&vb>0)||(va>0&&vb<0)) clipped.push_back(a+(b-a)*(va/(va-vb)));
    }
    polygon=std::move(clipped);
    if(polygon.size()<3) return false;
  }
  Kernel::Vector_3 sum=CGAL::NULL_VECTOR;
  for(const Point& p:polygon) sum=sum+(p-CGAL::ORIGIN);
  Point center=CGAL::ORIGIN+sum/Kernel::FT(polygon.size());
  for(const auto& plane:planes) if(plane.oriented_side(center)!=CGAL::ON_NEGATIVE_SIDE) return false;
  return true;
}

std::vector<std::array<std::int64_t,4>> export_faces(
    const Mesh& mesh,const std::array<Point,4>& original,Registry& registry) {
  std::array<Kernel::Plane_3,4> planes;
  for(int i=0;i<4;++i) planes[i]=Kernel::Plane_3(original[tetra_faces[i][0]],
      original[tetra_faces[i][1]],original[tetra_faces[i][2]]);
  std::vector<std::array<std::int64_t,4>> result;
  for(Face f:mesh.faces()) {
    std::array<Point,3> p;
    int n=0;
    for(Vertex v:CGAL::vertices_around_face(mesh.halfedge(f),mesh)) {
      if(n==3) throw std::runtime_error("The exact Boolean output is not triangulated.");
      p[n++]=mesh.point(v);
    }
    if(n!=3) throw std::runtime_error("Invalid Boolean polygon.");
    int source=-1;
    for(int i=0;i<4;++i)
      if(planes[i].has_on(p[0])&&planes[i].has_on(p[1])&&planes[i].has_on(p[2])) {
        source=i;break;
      }
    if(source<0) {
      auto region=mesh.property_map<Face,int>("f:region");
      if(!region.second||region.first[f]<1||region.first[f]>2)
        throw std::runtime_error("Lost cutter-region provenance in exact clipping.");
      source=3+region.first[f];
    }
    result.push_back({registry.add(p[0]),registry.add(p[1]),registry.add(p[2]),source});
  }
  return result;
}

int main(int argc,char**argv) {
  try {
    if(argc==4&&std::string(argv[1])=="--import") {
      std::ifstream in(argv[2],std::ios::binary);
      Mesh cutter=read_mesh(in);label_mesh(cutter,1);
      write_exact_mesh(argv[3],cutter);return 0;
    }
    if(argc==5&&std::string(argv[1])=="--prepare") {
      prepare(argv[2],argv[3],argv[4]);return 0;
    }
    if(argc!=5||std::string(argv[1])!="--clip")
      throw std::runtime_error("Usage: exact_basal_clip --prepare INPUT ARTIFACTS EXACT or --clip EXACT SOURCE OUTPUT");
    Mesh cutter=read_exact_mesh(argv[2]);
    Tree tree(cutter.faces().begin(),cutter.faces().end(),cutter);
    tree.build();tree.accelerate_distance_queries();
    CGAL::Side_of_triangle_mesh<Mesh,Kernel> side(tree);
    std::ifstream in(argv[3],std::ios::binary);
    if(!in) throw std::runtime_error("Cannot open exact-clip input.");
    std::cerr<<"[CGAL] cutter "<<num_vertices(cutter)<<" vertices, "<<num_faces(cutter)<<" triangles\n";
    Registry registry;
    const auto nv=read_scalar<std::int64_t>(in);
    for(std::int64_t i=0;i<nv;++i) {
      auto point=read_point(in);
      if(registry.add(point)!=i)
        throw std::runtime_error("Original mesh contains duplicate-coordinate point IDs.");
    }
    const auto count=read_scalar<std::int64_t>(in);
    std::vector<OutputPiece> output;
    output.reserve(count);
    auto start=std::chrono::steady_clock::now();
    for(std::int64_t index=0;index<count;++index) {
      const auto parent=read_scalar<std::int64_t>(in);
      std::array<Point,4> original;
      for(int j=0;j<4;++j) {
        auto id=read_scalar<std::int64_t>(in);
        if(id<0||id>=nv) throw std::runtime_error("Invalid source point ID.");
        original[j]=registry.points[id];
      }
      std::vector<Face> nearby;
      Kernel::Tetrahedron_3 query(original[0],original[1],original[2],original[3]);
      tree.all_intersected_primitives(query,std::back_inserter(nearby));
      bool crosses=false;
      for(Face f:nearby) if(triangle_crosses_interior(f,cutter,original)) {crosses=true;break;}
      if(!crosses) {
        Point center=CGAL::ORIGIN+((original[0]-CGAL::ORIGIN)+(original[1]-CGAL::ORIGIN)+
            (original[2]-CGAL::ORIGIN)+(original[3]-CGAL::ORIGIN))/4;
        auto where=side(center);
        if(where==CGAL::ON_BOUNDARY) throw std::runtime_error("Ambiguous exact uncut-cell classification.");
        OutputPiece piece;piece.parent=parent;piece.state=where==CGAL::ON_BOUNDED_SIDE?1:0;
        output.push_back(std::move(piece));continue;
      }
      Mesh tet=tetra_mesh(original), local=cutter, kept,removed;
      const auto before=positive_volume(tet);
      label_mesh(tet,0);
      auto success=booleans(tet,local,nullptr,&removed,&kept);
      if(!success[PMP::Corefinement::INTERSECTION]||!success[PMP::Corefinement::TM1_MINUS_TM2])
        throw std::runtime_error("Exact output is non-manifold at original parent "+std::to_string(parent));
      const auto kept_volume=positive_volume(kept),removed_volume=positive_volume(removed);
      if(kept_volume+removed_volume!=before)
        throw std::runtime_error("Exact parent-volume conservation failed at "+std::to_string(parent));
      OutputPiece piece;
      piece.parent=parent;
      piece.state=removed_volume==0?0:(kept_volume==0?1:2);
      if(piece.state==2) {
        piece.faces[0]=export_faces(kept,original,registry);
        piece.faces[1]=export_faces(removed,original,registry);
      }
      output.push_back(std::move(piece));
      if(index%100==0) {
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cerr<<"[CGAL] "<<index+1<<"/"<<count<<" parents, "<<seconds<<" s\n";
      }
    }
    std::ofstream out(argv[4],std::ios::binary);
    write_scalar<std::int64_t>(out,registry.points.size());
    for(const auto& p:registry.points) {
      write_scalar<double>(out,CGAL::to_double(p.x()));
      write_scalar<double>(out,CGAL::to_double(p.y()));
      write_scalar<double>(out,CGAL::to_double(p.z()));
    }
    write_scalar<std::int64_t>(out,output.size());
    for(const auto& piece:output) {
      write_scalar<std::int64_t>(out,piece.parent);
      write_scalar<std::int64_t>(out,piece.state);
      for(int side=0;side<2;++side) {
        write_scalar<std::int64_t>(out,piece.faces[side].size());
        for(const auto& face:piece.faces[side])
          for(auto id:face) write_scalar<std::int64_t>(out,id);
      }
    }
    if(!out) throw std::runtime_error("Cannot write exact-clip output.");
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"Exact basal clipping failed: "<<error.what()<<"\n";
    return 1;
  }
}
