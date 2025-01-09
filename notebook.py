import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md("""#3D Geometry File Formats""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## About STL

        STL is a simple file format which describes 3D objects as a collection of triangles.
        The acronym STL stands for "Simple Triangle Language", "Standard Tesselation Language" or "STereoLitography"[^1].

        [^1]: STL was invented for – and is still widely used – for 3D printing.
        """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/teapot.stl", theta=45.0, phi=30.0, scale=2))
    return


@app.cell
def __(mo):
    with open("data/teapot.stl", mode="rt", encoding="utf-8") as _file:
        teapot_stl = _file.read()

    teapot_stl_excerpt = teapot_stl[:723] + "..." + teapot_stl[-366:]

    mo.md(
        f"""
    ## STL ASCII Format

    The `data/teapot.stl` file provides an example of the STL ASCII format. It is quite large (more than 60000 lines) and looks like that:
    """
    +
    f"""```
    {teapot_stl_excerpt}
    ```
    """
    +

    """
    """
    )
    return teapot_stl, teapot_stl_excerpt


@app.cell
def __(mo):
    mo.md(f"""

      - Study the [{mo.icon("mdi:wikipedia")} STL (file format)](https://en.wikipedia.org/wiki/STL_(file_format)) page (or other online references) to become familiar the format.

      - Create a STL ASCII file `"data/cube.stl"` that represents a cube of unit length  
        (💡 in the simplest version, you will need 12 different facets).

      - Display the result with the function `show` (make sure to check different angles).
    """)
    return


@app.cell
def __(show):
    #réponse
    def cube_stl(filename):
        faces = [
            ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 0, -1]),
            ([(1, 0, 0), (1, 1, 0), (0, 1, 0)], [0, 0, -1]),

            ([(0, 0, 1), (0, 1, 1), (1, 0, 1)], [0, 0, 1]),
            ([(1, 0, 1), (0, 1, 1), (1, 1, 1)], [0, 0, 1]),

            ([(0, 0, 0), (0, 0, 1), (1, 0, 0)], [0, -1, 0]),
            ([(1, 0, 0), (0, 0, 1), (1, 0, 1)], [0, -1, 0]),

            ([(0, 1, 0), (1, 1, 0), (0, 1, 1)], [0, 1, 0]),
            ([(1, 1, 0), (1, 1, 1), (0, 1, 1)], [0, 1, 0]),

            ([(0, 0, 0), (0, 1, 0), (0, 0, 1)], [-1, 0, 0]),
            ([(0, 1, 0), (0, 1, 1), (0, 0, 1)], [-1, 0, 0]),

            ([(1, 0, 0), (1, 0, 1), (1, 1, 0)], [1, 0, 0]),
            ([(1, 1, 0), (1, 0, 1), (1, 1, 1)], [1, 0, 0]),]
        
        # on a bien 12 faces différentes
        
        with open(filename, 'w') as file:
            file.write("solid cube\n")
            for vertices, normal in faces:
                file.write(f"facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                file.write("outer loop\n")
                for vertex in vertices:
                    file.write(f"vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                file.write("endloop\n")
                file.write("endfacet\n")
            file.write("endsolid cube\n")

    cube_stl("data/cube.stl")

    show("data/cube.stl", theta=45.0, phi=30.0, scale=1.0) #on le voit de dessus

    return (cube_stl,)


@app.cell
def __(show):
    show("data/cube.stl", theta=90.0, phi=60.0, scale=1.0) # de coté
    return


@app.cell
def __(show):
    show("data/cube.stl", theta=120.0, phi=30.0, scale=1.0) 
    # on a envisagé suffisamment d'angles pour voir que le résultat est bien un cube
    return


@app.cell
def __(mo):
    mo.md(r"""## STL & NumPy""")
    return


@app.cell
def __(mo):
    mo.md(rf"""

    ### NumPy to STL

    Implement the following function:

    ```python
    def make_STL(triangles, normals=None, name=""):
        pass # 🚧 TODO!
    ```

    #### Parameters

      - `triangles` is a NumPy array of shape `(n, 3, 3)` and data type `np.float32`,
         which represents a sequence of `n` triangles (`triangles[i, j, k]` represents 
         is the `k`th coordinate of the `j`th point of the `i`th triangle)

      - `normals` is a NumPy array of shape `(n, 3)` and data type `np.float32`;
         `normals[i]` represents the outer unit normal to the `i`th facet.
         If `normals` is not specified, it should be computed from `triangles` using the 
         [{mo.icon("mdi:wikipedia")} right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule).

      - `name` is the (optional) solid name embedded in the STL ASCII file.

    #### Returns

      - The STL ASCII description of the solid as a string.

    #### Example

    Given the two triangles that make up a flat square:

    ```python

    square_triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    ```

    then printing `make_STL(square_triangles, name="square")` yields
    ```
    solid square
      facet normal 0.0 0.0 1.0
        outer loop
          vertex 0.0 0.0 0.0
          vertex 1.0 0.0 0.0
          vertex 0.0 1.0 0.0
        endloop
      endfacet
      facet normal 0.0 0.0 1.0
        outer loop
          vertex 1.0 1.0 0.0
          vertex 0.0 1.0 0.0
          vertex 1.0 0.0 0.0
        endloop
      endfacet
    endsolid square
    ```

    """)
    return


@app.cell
def __(np):
    #réponse
    def normales(v1, v2, v3):
        u = v2 - v1 # on définit 2 des vecteurs du triangle 
        v = v3 - v1
        normal = np.cross(u, v) # produit vectoriel bien orienté pour la normale
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 0 else np.array([0.0, 0.0, 0.0], dtype=np.float32) 
        # ces deux dernières lignes empêchent une erreur
        
    def make_STL(triangles, normals=None, name=""):
      
        if normals is None:
            normals = np.array([
                normales(triangle[0], triangle[1], triangle[2])
                for triangle in triangles], dtype=np.float32) # on calcule les normales de chaque triangle qui sont égales à None par défaut

        stl_lines = [f"solid {name}"]
        for normal, triangle in zip(normals, triangles):
            stl_lines.append(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}")
            stl_lines.append("outer loop")
            for vertex in triangle:
                stl_lines.append(f"vertex {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")
            stl_lines.append("endloop")
            stl_lines.append("endfacet")
        stl_lines.append(f"endsolid {name}")

        return "\n".join(stl_lines) # essentiel pour mettre toutes les lignes ensemble

    # On teste sur le carré
    if __name__ == "__main__": 
        
        carre = np.array(
        [  [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],],
        dtype=np.float32,)

        stl_string = make_STL(carre, name="square")
        print(stl_string)
    return carre, make_STL, normales, stl_string


@app.cell
def __():
    # on a exactement le rendu souhaité
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### STL to NumPy

        Implement a `tokenize` function


        ```python
        def tokenize(stl):
            pass # 🚧 TODO!
        ```

        that is consistent with the following documentation:


        #### Parameters

          - `stl`: a Python string that represents a STL ASCII model.

        #### Returns

          - `tokens`: a list of STL keywords (`solid`, `facet`, etc.) and `np.float32` numbers.

        #### Example

        For the ASCII representation the square `data/square.stl`, printing the tokens with

        ```python
        with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
            square_stl = square_file.read()
        tokens = tokenize(square_stl)
        print(tokens)
        ```

        yields

        ```python
        ['solid', 'square', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(0.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'endloop', 'endfacet', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(1.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'endloop', 'endfacet', 'endsolid', 'square']
        ```
        """
    )
    return


@app.cell
def __(np):
    import re

    def tokenize(stl):
        # expression régulière
        pattern = r'\b(?:solid|facet|normal|outer|loop|vertex|endloop|endfacet|endsolid)\b|[-+]?\d*\.\d+|\d+'

        # Trouver tous les matches dans le fichier STL
        matches = re.findall(pattern, stl)

        tokens = []
        for match in matches:
            if match in {"solid", "facet", "normal", "outer", "loop", "vertex", "endloop", "endfacet", "endsolid"}:
                tokens.append(match)
            else:
                tokens.append(np.float32(match))

        return tokens
    return re, tokenize


@app.cell
def __(tokenize):
    with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
        square_stl = square_file.read()
    tokens = tokenize(square_stl)
    print(tokens)
    return square_file, square_stl, tokens


@app.cell
def __(mo):
    mo.md(
        """
        Implement a `parse` function


        ```python
        def parse(tokens):
            pass # 🚧 TODO!
        ```

        that is consistent with the following documentation:


        #### Parameters

          - `tokens`: a list of tokens

        #### Returns

        A `triangles, normals, name` triple where

          - `triangles`: a `(n, 3, 3)` NumPy array with data type `np.float32`,

          - `normals`: a `(n, 3)` NumPy array with data type `np.float32`,

          - `name`: a Python string.

        #### Example

        For the ASCII representation `square_stl` of the square,
        tokenizing then parsing

        ```python
        with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
            square_stl = square_file.read()
        tokens = tokenize(square_stl)
        triangles, normals, name = parse(tokens)
        print(repr(triangles))
        print(repr(normals))
        print(repr(name))
        ```

        yields

        ```python
        array([[[0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.]],

               [[1., 1., 0.],
                [0., 1., 0.],
                [1., 0., 0.]]], dtype=float32)
        array([[0., 0., 1.],
               [0., 0., 1.]], dtype=float32)
        'square'
        ```
        """
    )
    return


@app.cell
def __(np):

    def parse(tokens):
        
        triangles = []
        normals = []
        name = None

        # on parcourt les tokens
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == "solid":
                name = tokens[i + 1]
                i += 2

            elif token == "facet":
                normal = [float(tokens[i + 2]), float(tokens[i + 3]), float(tokens[i + 4])]
                normals.append(normal)
                i += 5  # On saute "facet normal nx ny nz"

                assert tokens[i] == "outer" and tokens[i + 1] == "loop"
                i += 2  # On saute "outer loop"

                vertices = []
                for _ in range(3):
                    assert tokens[i] == "vertex"
                    vertex = [float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])]
                    vertices.append(vertex)
                    i += 4 

                triangles.append(vertices)

                assert tokens[i] == "endloop"
                i += 1 

                assert tokens[i] == "endfacet"
                i += 1 

            elif token == "endsolid":
                i += 1

            else:
                i+=1 # au cas où ça plante,si on rencontre un autre token, comme c'est arrivé pour square, lorsque je n'avais pas mis cette commande


        triangles = np.array(triangles, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)

        return triangles, normals, name

        
    return (parse,)


@app.cell
def __(parse, square_stl, tokenize):

    tokens_ = tokenize(square_stl)
    triangles, normals, name = parse(tokens_)
    print(repr(triangles))
    print(repr(normals))
    print(repr(name))
    return name, normals, tokens_, triangles


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## Rules & Diagnostics



        Make diagnostic functions that check whether a STL model satisfies the following rules

          - **Positive octant rule.** All vertex coordinates are non-negative.

          - **Orientation rule.** All normals are (approximately) unit vectors and follow the [{mo.icon("mdi:wikipedia")} right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule).

          - **Shared edge rule.** Each triangle edge appears exactly twice.

          - **Ascending rule.** the z-coordinates of (the barycenter of) each triangle are a non-decreasing sequence.

    When the rule is broken, make sure to display some sensible quantitative measure of the violation (in %).

    For the record, the `data/teapot.STL` file:

      - 🔴 does not obey the positive octant rule,
      - 🟠 almost obeys the orientation rule, 
      - 🟢 obeys the shared edge rule,
      - 🔴 does not obey the ascending rule.

    Check that your `data/cube.stl` file does follow all these rules, or modify it accordingly!

    """
    )
    return


@app.cell
def __(np):
    def positive_octant_rule(triangles):
        violations = np.sum(triangles < 0)
        total_vertices = triangles.size
        return (violations / total_vertices) * 100

    def orientation_rule(triangles, normals):
        
        m = np.linalg.norm(normals, axis=1)
        unit_violations = np.sum(np.abs(m - 1) > 1e-6) #précision arbitraire

        # Right hand rule :
        calculated_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        calculated_normals /= np.linalg.norm(calculated_normals, axis=1, keepdims=True)
        right_hand_violations = np.sum(np.abs(calculated_normals - normals) > 1e-6)

        return (unit_violations / len(normals)) * 100, (right_hand_violations / len(triangles)) * 100

    def shared_edge_rule(triangles):
        edges = []
        for triangle in triangles:
            edges.extend([(tuple(triangle[i]), tuple(triangle[(i+1) % 3])) for i in range(3)])

        # On crée un dico où chaque clé est une arête, et chaque valeur est le nombre de fois où cette arête apparaît dans la liste edges.
        
        edges = [tuple(sorted(edge)) for edge in edges]
        edge_counts = {edge: edges.count(edge) for edge in edges}
        violations = sum(1 for count in edge_counts.values() if count != 2)

        total_edges = len(edges)
        return (violations / total_edges) * 100

    def ascending_rule(triangles):
        barycentres = np.mean(triangles, axis=1)
        z_coordinates = barycentres[:, 2]
        violations = np.sum(np.diff(z_coordinates) < 0)
        return (violations / (len(z_coordinates) - 1)) * 100
    return (
        ascending_rule,
        orientation_rule,
        positive_octant_rule,
        shared_edge_rule,
    )


@app.cell
def __(
    ascending_rule,
    orientation_rule,
    parse,
    positive_octant_rule,
    shared_edge_rule,
    tokenize,
):
    def obey(stl_file): 
        tokens = tokenize(stl_file)
        triangles, normals, name = parse(tokens)
        violations = {
            "Positive Octant Rule": positive_octant_rule(triangles),
            "Orientation Rule (Unit Normals)": orientation_rule(triangles, normals)[0],
            "Orientation Rule (Right-Hand Rule)": orientation_rule(triangles, normals)[1],
            "Shared Edge Rule": shared_edge_rule(triangles),
            "Ascending Rule": ascending_rule(triangles),
        }

        return violations
       
    return (obey,)


@app.cell
def __(obey):
    with open("data/cube.stl", mode="rt", encoding="utf-8") as stl_file:        stl_content = stl_file.read()
    violations = obey(stl_content)
    print("Résultats des vérifications :", violations)


    return stl_content, stl_file, violations


@app.cell
def __(mo):
    mo.md(
    rf"""
    ## OBJ Format

    The OBJ format is an alternative to the STL format that looks like this:

    ```
    # OBJ file format with ext .obj
    # vertex count = 2503
    # face count = 4968
    v -3.4101800e-003 1.3031957e-001 2.1754370e-002
    v -8.1719160e-002 1.5250145e-001 2.9656090e-002
    v -3.0543480e-002 1.2477885e-001 1.0983400e-003
    v -2.4901590e-002 1.1211138e-001 3.7560240e-002
    v -1.8405680e-002 1.7843055e-001 -2.4219580e-002
    ...
    f 2187 2188 2194
    f 2308 2315 2300
    f 2407 2375 2362
    f 2443 2420 2503
    f 2420 2411 2503
    ```

    This content is an excerpt from the `data/bunny.obj` file.

    """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/bunny.obj", scale="1.5"))
    return


@app.cell
def __(mo):
    mo.md(
        """
        Study the specification of the OBJ format (search for suitable sources online),
        then develop a `OBJ_to_STL` function that is rich enough to convert the OBJ bunny file into a STL bunny file.
        """
    )
    return


@app.cell
def __(np, os):
    def OBJ_to_STL(obj_filename, stl_filename):
        vertices = []
        faces = []

        with open(obj_filename, 'r') as obj_file:
            for line in obj_file:
                parts = line.strip().split()
                if not parts or parts[0].startswith('#'):
                    continue  # On skip ce qui est inutile
                if parts[0] == 'v':
                    vertex = list(map(float, parts[1:4]))
                    vertices.append(vertex)
                elif parts[0] == 'f':
                    face = [int(idx.split('/')[0]) - 1 for idx in parts[1:4]]
                    faces.append(face)

    # A noter que je n'ai pas pris en compte les vt qui ne sont pas utiles ici

                    
        os.makedirs(os.path.dirname(stl_filename), exist_ok=True)
        with open(stl_filename, 'w') as stl_file:
            stl_file.write(f"solid model\n")
            for face in faces:
                v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

                vec1 = [v2[i] - v1[i] for i in range(3)]
                vec2 = [v3[i] - v1[i] for i in range(3)]
                normal = np.cross(vec1, vec2)
                norm = np.linalg.norm(normal)
                if norm != 0:
                    normal = normal / norm

                stl_file.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                stl_file.write(f"outer loop\n")
                stl_file.write(f"vertex {v1[0]} {v1[1]} {v1[2]}\n")
                stl_file.write(f"vertex {v2[0]} {v2[1]} {v2[2]}\n")
                stl_file.write(f"vertex {v3[0]} {v3[1]} {v3[2]}\n")
                stl_file.write(f"endloop\n")
                stl_file.write(f"endfacet\n")
            stl_file.write(f"endsolid model\n")

        print(f"Converted {obj_filename} to {stl_filename} successfully.")

    OBJ_to_STL("data/bunny.obj","data/bunny.stl")
    return (OBJ_to_STL,)


@app.cell
def __(show):
    show("data/bunny.stl")
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## Binary STL

    Since the STL ASCII format can lead to very large files when there is a large number of facets, there is an alternate, binary version of the STL format which is more compact.

    Read about this variant online, then implement the function

    ```python
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        pass  # 🚧 TODO!
    ```

    that will convert a binary STL file to a ASCII STL file. Make sure that your function works with the binary `data/dragon.stl` file which is an example of STL binary format.

    💡 The `np.fromfile` function may come in handy.

        """
    )
    return


@app.cell
def __(np):
    import os
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        with open(stl_filename_in, 'rb') as binary_file:
            # Lit le header de 80 bits
            header = binary_file.read(80)

            # nombre de triangles
            num_triangles = np.fromfile(binary_file, dtype=np.uint32, count=1)[0]

            facets = []

            for _ in range(num_triangles):
                
                # Vecteur normal (3 floats)
                normal = np.fromfile(binary_file, dtype=np.float32, count=3)

                vertices = np.fromfile(binary_file, dtype=np.float32, count=9).reshape(3, 3)

                # on skip le compte des bits
                binary_file.read(2)

                facets.append((normal, vertices))


        os.makedirs(os.path.dirname(stl_filename_out), exist_ok=True)
        
    # On écrit à présent le fichier en ASCII

        with open(stl_filename_out, 'w') as ascii_file:
            ascii_file.write("solid model\n")
            for normal, vertices in facets:
                ascii_file.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                ascii_file.write("    outer loop\n")
                for vertex in vertices:
                    ascii_file.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                ascii_file.write("    endloop\n")
                ascii_file.write("  endfacet\n")
            ascii_file.write("endsolid model\n")

        print(f"Converted binary STL {stl_filename_in} to ASCII STL {stl_filename_out}")

    STL_binary_to_text("data/dragon.stl", "dragon/output_ascii.stl")
    return STL_binary_to_text, os


@app.cell
def __():
    # Il y avait un message d'erreur avant que je n'utilise os, disant que le fichier output n'était pas créé
    # os et os.makedirs est la solution que j'ai trouvée pour éviter cela

    # on vérifie que le programme renvoie bien le dragon, avec un temps d'exécution supérieur au binaire ce qui est cohérent puisqu'il prend plus de place (défaut de ascii)
    return


@app.cell
def __(show):
    show("dragon/output_ascii.stl", theta=75.0, phi=-20.0, scale=1.7)
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/dragon.stl", theta=75.0, phi=-20.0, scale=1.7))
    return


@app.cell(hide_code=True)
def __(make_STL, np):
    def STL_binary_to_text2(stl_filename_in, stl_filename_out):
        with open(stl_filename_in, mode="rb") as file:
            _ = file.read(80)
            n = np.fromfile(file, dtype=np.uint32, count=1)[0]
            normals = []
            faces = []
            for i in range(n):
                normals.append(np.fromfile(file, dtype=np.float32, count=3))
                faces.append(np.fromfile(file, dtype=np.float32, count=9).reshape(3, 3))
                _ = file.read(2)
        stl_text = make_STL(faces, normals)
        with open(stl_filename_out, mode="wt", encoding="utf-8") as file:
            file.write(stl_text)
    return (STL_binary_to_text2,)


@app.cell
def __(mo):
    mo.md(rf"""## Constructive Solid Geometry (CSG)

    Have a look at the documentation of [{mo.icon("mdi:github")}fogleman/sdf](https://github.com/fogleman/) and study the basics. At the very least, make sure that you understand what the code below does:
    """)
    return


@app.cell
def __(X, Y, Z, box, cylinder, mo, show, sphere):
    demo_csg = sphere(1) & box(1.5)
    _c = cylinder(0.5)
    demo_csg = demo_csg - (_c.orient(X) | _c.orient(Y) | _c.orient(Z))
    demo_csg.save('output/demo-csg.stl', step=0.05)
    mo.show_code(show("output/demo-csg.stl", theta=45.0, phi=45.0, scale=1.0))
    return (demo_csg,)


@app.cell
def __(mo):
    mo.md("""ℹ️ **Remark.** The same result can be achieved in a more procedural style, with:""")
    return


@app.cell
def __(
    box,
    cylinder,
    difference,
    intersection,
    mo,
    orient,
    show,
    sphere,
    union,
):
    demo_csg_alt = difference(
        intersection(
            sphere(1),
            box(1.5),
        ),
        union(
            orient(cylinder(0.5), [1.0, 0.0, 0.0]),
            orient(cylinder(0.5), [0.0, 1.0, 0.0]),
            orient(cylinder(0.5), [0.0, 0.0, 1.0]),
        ),
    )
    demo_csg_alt.save("output/demo-csg-alt.stl", step=0.05)
    mo.show_code(show("output/demo-csg-alt.stl", theta=45.0, phi=45.0, scale=1.0))
    return (demo_csg_alt,)


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## JupyterCAD

    [JupyterCAD](https://github.com/jupytercad/JupyterCAD) is an extension of the Jupyter lab for 3D geometry modeling.

      - Use it to create a JCAD model that correspond closely to the `output/demo_csg` model;
    save it as `data/demo_jcad.jcad`.

      - Study the format used to represent JupyterCAD files (💡 you can explore the contents of the previous file, but you may need to create some simpler models to begin with).

      - When you are ready, create a `jcad_to_stl` function that understand enough of the JupyterCAD format to convert `"data/demo_jcad.jcad"` into some corresponding STL file.
    (💡 do not tesselate the JupyterCAD model by yourself, instead use the `sdf` library!)


        """
    )
    return


@app.cell
def __(box, cylinder, difference, intersection, json, os, sphere, union):

    def jcad_to_stl(jcad_filename, stl_filename):
       
        os.makedirs(os.path.dirname(stl_filename), exist_ok=True)

        with open(jcad_filename, 'r') as file:
            jcad_data = json.load(file)

        shapes = {}
        for shape in jcad_data.get("shapes", []):
            shape_type = shape.get("type")
            dimensions = shape.get("dimensions", {})
            position = shape.get("position")
            rotation = shape.get("rotation")

            # On aura besoin uniquement de ces 3 formes
            if shape_type == "sphere":
                radius = dimensions.get("radius")
                obj = sphere(radius)
            elif shape_type == "box":
                size = dimensions.get("size")
                obj = box(size)
            elif shape_type == "cylinder":
                radius = dimensions.get("radius")
                height = dimensions.get("height")
                obj = cylinder(radius, height)

            obj = obj.translate(position)

            shapes[shape["id"]] = obj

        for operation in jcad_data.get("operations", []):
            op_type = operation.get("type")
            op_shapes = operation.get("shapes", [])
            op_id = operation.get("id")

            if op_type == "union":
                result = union(*[shapes[_] for _ in op_shapes])
            elif op_type == "intersection":
                result = intersection(*[shapes[_] for _ in op_shapes])
            elif op_type == "difference":
                result = difference(*[shapes[_] for _ in op_shapes])
            shapes[op_id] = result

        final_model_id = jcad_data.get("final_model", None)
        final_model = shapes[final_model_id]
        final_model.save(stl_filename)
        print(f"Converted {jcad_filename} to {stl_filename}")
    return (jcad_to_stl,)


@app.cell
def __(json, os):

    demo_csg_jcad = {
        "name": "demo_csg_model",
        "shapes": [
            {
                "id": "sphere1",
                "type": "sphere",
                "dimensions": {"radius": 1.0},
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
            },
            {
                "id": "box1",
                "type": "box",
                "dimensions": {"size": [1.5, 1.5, 1.5]},
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
            },
            {
                "id": "cylinder_x",
                "type": "cylinder",
                "dimensions": {"radius": 0.5, "height": 3.0},
                "position": [0, 0, 0],
                "rotation": [90, 0, 0],
            },
            {
                "id": "cylinder_y",
                "type": "cylinder",
                "dimensions": {"radius": 0.5, "height": 3.0},
                "position": [0, 0, 0],
                "rotation": [0, 90, 0],
            },
            {
                "id": "cylinder_z",
                "type": "cylinder",
                "dimensions": {"radius": 0.5, "height": 3.0},
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
            },
        ],
        "operations": [
            {
                "type": "intersection",
                "shapes": ["sphere1", "box1"],
                "id": "intersect_sphere_box",
            },
            {
                "type": "union",
                "shapes": ["cylinder_x", "cylinder_y", "cylinder_z"],
                "id": "union_cylinders",
            },
            {
                "type": "difference",
                "shapes": ["intersect_sphere_box", "union_cylinders"],
                "id": "fig_finale",
            },
        ],
        "fig_finale": "fig_finale",}

    jcad_filename = "data/demo_csg.jcad"
    os.makedirs(os.path.dirname(jcad_filename), exist_ok=True)
    with open(jcad_filename, "w") as jcad_file:
        json.dump(demo_csg_jcad, jcad_file)

    print(f"JCAD file : {jcad_filename}")

    return demo_csg_jcad, jcad_file, jcad_filename


@app.cell(hide_code=True)
def __(jcad_to_stl, show):
    jcad_to_stl("data/demo_csg.jcad", "output/demo_csg.stl")
    show("output/demo_csg.stl", theta=45.0, phi=45.0, scale=1.0)

    return


@app.cell
def __(mo):
    mo.md("""## Appendix""")
    return


@app.cell
def __(mo):
    mo.md("""### Dependencies""")
    return


@app.cell
def __():
    # Python Standard Library
    import json

    # Marimo
    import marimo as mo

    # Third-Party Librairies
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl3d
    from mpl3d import glm
    from mpl3d.mesh import Mesh
    from mpl3d.camera import Camera

    import meshio

    np.seterr(over="ignore")  # 🩹 deal with a meshio false warning

    import sdf
    from sdf import sphere, box, cylinder
    from sdf import X, Y, Z
    from sdf import intersection, union, orient, difference

    mo.show_code()
    return (
        Camera,
        Mesh,
        X,
        Y,
        Z,
        box,
        cylinder,
        difference,
        glm,
        intersection,
        json,
        meshio,
        mo,
        mpl3d,
        np,
        orient,
        plt,
        sdf,
        sphere,
        union,
    )


@app.cell
def __(mo):
    mo.md(r"""### STL Viewer""")
    return


@app.cell
def __(Camera, Mesh, glm, meshio, mo, plt):
    def show(
        filename,
        theta=0.0,
        phi=0.0,
        scale=1.0,
        colormap="viridis",
        edgecolors=(0, 0, 0, 0.25),
        figsize=(6, 6),
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1)
        ax.axis("off")
        camera = Camera("ortho", theta=theta, phi=phi, scale=scale)
        mesh = meshio.read(filename)
        vertices = glm.fit_unit_cube(mesh.points)
        faces = mesh.cells[0].data
        vertices = glm.fit_unit_cube(vertices)
        mesh = Mesh(
            ax,
            camera.transform,
            vertices,
            faces,
            cmap=plt.get_cmap(colormap),
            edgecolors=edgecolors,
        )
        return mo.center(fig)

    mo.show_code()
    return (show,)


@app.cell
def __(mo, show):
    mo.show_code(show("data/teapot.stl", theta=45.0, phi=30.0, scale=2))
    return


if __name__ == "__main__":
    app.run()
