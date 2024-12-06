package org.example.service.impl;

import org.example.dao.StudentDao;
import org.example.entity.Student;
import org.example.service.StudentService;
import org.springframework.stereotype.Component;


import java.util.List;

@Component
public class StudentServiceImpl implements StudentService {

    private StudentDao studentDao;

    public void setStudentDao(StudentDao studentDao) {
        this.studentDao = studentDao;
    }

    public StudentDao getStudentDao() {
        return studentDao;
    }

    @Override
    public List<Student> findAll() {
        return studentDao.selectAll();
    }

    @Override
    public void add_data(Student student) {
        studentDao.insert(student);
    }

    @Override
    public int remove(int id) {
        return studentDao.delete(id);
    }

    @Override
    public Student searchById(int id) {
        Student student  =  studentDao.findById(id);
        System.out.println(student);
        return student;
    }


}
