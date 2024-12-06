package org.example.dao;


import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.example.entity.Student;
import org.springframework.stereotype.Repository;

import java.util.List;


@Repository
public interface StudentDao {
   public List<Student> selectAll();
   public void insert (Student student);
   public int delete(int id);
   public Student findById(@Param("id") int id);
}
