use nalgebra::Vector3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridToObject {
    pub grid_index: usize,
    pub object_index: usize,
}

impl PartialOrd for GridToObject {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GridToObject {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.grid_index.cmp(&other.grid_index)
    }
}

pub struct SearchGrid {
    pub grid_to_object: Vec<GridToObject>,
    pub grid_to_object_index: Vec<usize>,
    pub cell_size: f32,
    pub bounding_box_min: Vector3<f32>,
    pub bounding_box_max: Vector3<f32>,
    pub dimensions: Vector3<usize>,
}

impl SearchGrid {
    pub fn new() -> Self {
        Self {
            grid_to_object: Vec::new(),
            grid_to_object_index: Vec::new(),
            cell_size: 1.0,
            bounding_box_min: Vector3::zeros(),
            bounding_box_max: Vector3::repeat(1.0),
            dimensions: Vector3::repeat(1),
        }
    }

    pub fn initialize(&mut self, bb_min: Vector3<f32>, bb_max: Vector3<f32>, cell_size: f32, num_objects: usize) {
        self.cell_size = cell_size;
        self.bounding_box_min = bb_min;
        self.bounding_box_max = bb_max;
        self.dimensions = Vector3::new(
            ((bb_max.x - bb_min.x) / cell_size).ceil() as usize,
            ((bb_max.y - bb_min.y) / cell_size).ceil() as usize,
            ((bb_max.z - bb_min.z) / cell_size).ceil() as usize,
        );
        self.grid_to_object = vec![
            GridToObject {
                grid_index: 0,
                object_index: 0
            };
            num_objects
        ];
    }

    pub fn grid_index(&self, point: Vector3<f32>) -> Vector3<usize> {
        let indices = (point - self.bounding_box_min)
            .map(|v| (v / self.cell_size).floor() as isize)
            .map(|v| v.max(0).min(self.dimensions.x as isize - 1));
        indices.map(|x| x as usize)
    }

    pub fn get_flattened_index(&self, point: Vector3<f32>) -> usize {
        let indices = self.grid_index(point);
        indices.z * self.dimensions.y * self.dimensions.x
            + indices.y * self.dimensions.x
            + indices.x
    }

    pub fn post_process(&mut self, is_initial: bool) {
        if is_initial {
            self.grid_to_object.sort();
        } else {
            self.grid_to_object.sort_by(|a, b| a.grid_index.cmp(&b.grid_index));
        }

        let num_cells = self.dimensions.x * self.dimensions.y * self.dimensions.z;
        self.grid_to_object_index = vec![0; num_cells + 1];
        let mut i = 0;
        for cell in 0..num_cells {
            while i < self.grid_to_object.len() && self.grid_to_object[i].grid_index <= cell {
                i += 1;
            }
            self.grid_to_object_index[cell + 1] = i;
        }
    }

    pub fn get_one_ring_neighbors(&self, point: Vector3<f32>) -> Vec<usize> {
        let indices = self.grid_index(point);
        let mut neighbors = Vec::new();

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let x = (indices.x as isize + dx).max(0).min(self.dimensions.x as isize - 1) as usize;
                    let y = (indices.y as isize + dy).max(0).min(self.dimensions.y as isize - 1) as usize;
                    let z = (indices.z as isize + dz).max(0).min(self.dimensions.z as isize - 1) as usize;

                    let cell_index = z * self.dimensions.y * self.dimensions.x
                        + y * self.dimensions.x
                        + x;

                    for i in self.grid_to_object_index[cell_index]
                        ..self.grid_to_object_index[cell_index + 1]
                    {
                        neighbors.push(self.grid_to_object[i].object_index);
                    }
                }
            }
        }

        neighbors
    }
}
