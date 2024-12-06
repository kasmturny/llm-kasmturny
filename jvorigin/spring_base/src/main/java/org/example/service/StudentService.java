package org.example.service;

import org.example.entity.Student;

import java.util.List;


public interface StudentService {
    public List<Student> findAll();
    public void add_data(Student student);
    public int remove(int id);
    public Student searchById(int id);
}
