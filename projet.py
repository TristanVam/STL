import marimo

__generated_with = "0.9.17"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import numpy as np
    return (np,)


@app.cell
def __():
    def write_cube_stl(filename):
        cube_facets = [
            # Bottom face
            ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 0, -1]),
            ([(1, 0, 0), (1, 1, 0), (0, 1, 0)], [0, 0, -1]),
            # Top face
            ([(0, 0, 1), (0, 1, 1), (1, 0, 1)], [0, 0, 1]),
            ([(1, 0, 1), (0, 1, 1), (1, 1, 1)], [0, 0, 1]),
            # Front face
            ([(0, 0, 0), (0, 0, 1), (1, 0, 0)], [0, -1, 0]),
            ([(1, 0, 0), (0, 0, 1), (1, 0, 1)], [0, -1, 0]),
            # Back face
            ([(0, 1, 0), (1, 1, 0), (0, 1, 1)], [0, 1, 0]),
            ([(1, 1, 0), (1, 1, 1), (0, 1, 1)], [0, 1, 0]),
            # Left face
            ([(0, 0, 0), (0, 1, 0), (0, 0, 1)], [-1, 0, 0]),
            ([(0, 1, 0), (0, 1, 1), (0, 0, 1)], [-1, 0, 0]),
            # Right face
            ([(1, 0, 0), (1, 0, 1), (1, 1, 0)], [1, 0, 0]),
            ([(1, 1, 0), (1, 0, 1), (1, 1, 1)], [1, 0, 0]),
        ]

        with open(filename, 'w') as file:
            file.write("solid cube\n")
            for vertices, normal in cube_facets:
                file.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                file.write("    outer loop\n")
                for vertex in vertices:
                    file.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                file.write("    endloop\n")
                file.write("  endfacet\n")
            file.write("endsolid cube\n")

    write_cube_stl("data/cube.stl")

    return (write_cube_stl,)


@app.cell
def __(np):

    def compute_normal(v1, v2, v3):
        """Compute the unit normal vector using the right-hand rule."""
        u = v2 - v1
        v = v3 - v1
        normal = np.cross(u, v)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 0 else np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def make_STL(triangles, normals=None, name=""):
        """
        Generate the ASCII STL representation of a solid.

        Parameters:
            triangles: NumPy array of shape (n, 3, 3) and dtype np.float32.
            normals: Optional NumPy array of shape (n, 3) and dtype np.float32.
            name: Optional name of the solid.

        Returns:
            STL ASCII representation as a string.
        """
        if normals is None:
            normals = np.array([
                compute_normal(triangle[0], triangle[1], triangle[2])
                for triangle in triangles
            ], dtype=np.float32)

        stl_lines = [f"solid {name}"]
        for normal, triangle in zip(normals, triangles):
            stl_lines.append(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}")
            stl_lines.append("    outer loop")
            for vertex in triangle:
                stl_lines.append(f"      vertex {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")
            stl_lines.append("    endloop")
            stl_lines.append("  endfacet")
        stl_lines.append(f"endsolid {name}")

        return "\n".join(stl_lines)

    # Example Usage
    if __name__ == "__main__":
        # Define two triangles representing a square
        square_triangles = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ], dtype=np.float32)

        # Generate STL ASCII string
        stl_string = make_STL(square_triangles, name="square")
        print(stl_string)
    return compute_normal, make_STL, square_triangles, stl_string


@app.cell
def __():
    #compute_normal Fonction : Calcule le vecteur normal d'un triangle en utilisant le produit en croix et le normalise à la longueur unitaire.
    #Validation des entrées : Si les normales ne sont pas fournies, elles sont calculées à partir des triangles à l'aide de compute_normal.
    #Génération du format STL : Chaque triangle et sa normale correspondante sont convertis au format ASCII STL.
     #Sortie : Renvoie la représentation STL ASCII de la chaîne de caractères du solide.
    return


@app.cell
def __(np):

    import re

    def tokenize(stl):
        """
        Tokenizes an STL ASCII string into a list of keywords and np.float32 numbers.

        Parameters:
            stl (str): A Python string representing an STL ASCII model.

        Returns:
            list: A list containing STL keywords and np.float32 numbers.
        """
        # Regular expression to match STL keywords or floating-point numbers
        pattern = r'\b(?:solid|facet|normal|outer|loop|vertex|endloop|endfacet|endsolid)\b|[-+]?\d*\.\d+|\d+'
        
        # Find all matches in the STL string
        matches = re.findall(pattern, stl)
        
        tokens = []
        for match in matches:
            if match in {"solid", "facet", "normal", "outer", "loop", "vertex", "endloop", "endfacet", "endsolid"}:
                # Append keywords directly as strings
                tokens.append(match)
            else:
                # Convert numbers to np.float32
                tokens.append(np.float32(match))
        
        return tokens

    return re, tokenize


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
